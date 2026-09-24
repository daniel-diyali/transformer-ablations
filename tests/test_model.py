"""Model specification.

The two tests that gate this milestone are attention equivalence against
PyTorch's fused kernel and causality by perturbation. Everything else guards
a property some ablation condition depends on.

Configs are deliberately tiny so the whole file runs on CPU in seconds.
"""

import math
from itertools import product

import pytest
import torch
import torch.nn.functional as F

from minigpt.model import (
    GPT,
    Block,
    CausalSelfAttention,
    GPTConfig,
    apply_rope,
    causal_attention,
    rope_tables,
    sinusoidal_encoding,
)

VOCAB = 128
BASE = dict(vocab_size=VOCAB, n_layer=2, n_head=4, d_model=32, block_size=16)

POS_ENCODINGS = ("learned", "sinusoidal", "rope")
NORM_PLACEMENTS = ("pre", "post")
ALL_CONDITIONS = list(product(POS_ENCODINGS, NORM_PLACEMENTS))


def build(**overrides) -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(**{**BASE, **overrides}))


# ------------------------------------------------------------- attention


def test_attention_matches_pytorchs_fused_kernel():
    """The gate: proves the hand-rolled path is correct, not merely plausible."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 8, dtype=torch.float64) for _ in range(3))

    mine = causal_attention(q, k, v)
    reference = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    assert torch.allclose(mine, reference, atol=1e-12)


def test_attention_gives_masked_positions_exactly_zero_weight():
    """Masking before the softmax, not after. After would leave small weights."""
    torch.manual_seed(0)
    t = 8
    q, k = torch.randn(1, 1, t, 4), torch.randn(1, 1, t, 4)
    v = torch.eye(t).view(1, 1, t, t)

    # With v as identity, the output row IS the attention weight row.
    weights = causal_attention(q, k, v)[0, 0]

    assert torch.triu(weights, diagonal=1).abs().max() == 0.0
    assert torch.allclose(weights.sum(dim=-1), torch.ones(t))


def test_first_position_can_only_attend_to_itself():
    torch.manual_seed(0)
    q, k = torch.randn(1, 1, 5, 4), torch.randn(1, 1, 5, 4)
    weights = causal_attention(q, k, torch.eye(5).view(1, 1, 5, 5))[0, 0]
    assert weights[0, 0] == pytest.approx(1.0)


@pytest.mark.parametrize("pos_encoding,norm_placement", ALL_CONDITIONS)
def test_editing_a_token_cannot_change_earlier_logits(pos_encoding, norm_placement):
    """The gate: causality by perturbation.

    Catches a wrong mask and a correct mask applied wrongly. Inspecting the
    mask tensor catches only the first.
    """
    model = build(pos_encoding=pos_encoding, norm_placement=norm_placement).eval()
    idx = torch.randint(0, VOCAB, (1, 12))
    edited = idx.clone()
    edited[0, 6] = (idx[0, 6] + 1) % VOCAB

    with torch.no_grad():
        before, _ = model(idx)
        after, _ = model(edited)

    assert torch.equal(before[:, :6], after[:, :6]), "a later token changed an earlier logit"
    assert not torch.equal(before[:, 6:], after[:, 6:]), "the edit had no effect at all"


# ----------------------------------------------------- positional encoding


def test_rope_makes_attention_depend_only_on_relative_position():
    """The property RoPE exists for, and why it extrapolates."""
    head_dim, t = 16, 64
    cos, sin = rope_tables(t, head_dim)
    a, b = torch.randn(head_dim), torch.randn(head_dim)

    def score(m: int, n: int) -> float:
        qs, ks = torch.zeros(1, 1, t, head_dim), torch.zeros(1, 1, t, head_dim)
        qs[0, 0, m], ks[0, 0, n] = a, b
        return (apply_rope(qs, cos, sin)[0, 0, m] * apply_rope(ks, cos, sin)[0, 0, n]).sum().item()

    same_gap = [score(m, m - 2) for m in (5, 15, 40)]
    assert max(same_gap) - min(same_gap) < 1e-5
    assert abs(same_gap[0] - score(10, 3)) > 1e-3


def test_rope_preserves_vector_norm():
    """It is a rotation; anything else would rescale the residual stream."""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 8)
    cos, sin = rope_tables(32, 8)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_sinusoidal_table_is_deterministic():
    """No seed, no training: two builds must agree exactly."""
    assert torch.equal(sinusoidal_encoding(32, 16), sinusoidal_encoding(32, 16))


def test_sinusoidal_values_stay_bounded():
    table = sinusoidal_encoding(256, 64)
    assert table.min() >= -1.0 and table.max() <= 1.0


@pytest.mark.parametrize("d_model", [7, 8, 31, 32])
def test_sinusoidal_handles_odd_and_even_widths(d_model):
    assert sinusoidal_encoding(12, d_model).shape == (12, d_model)


# ----------------------------------------------- long-context behaviour (A2)


@pytest.mark.parametrize("pos_encoding", ["sinusoidal", "rope"])
def test_deterministic_encodings_run_beyond_the_training_context(pos_encoding):
    """What makes the A2 long-context arm possible at all."""
    model = build(pos_encoding=pos_encoding).eval()
    with torch.no_grad():
        logits, _ = model(torch.randint(0, VOCAB, (1, BASE["block_size"] * 2)))
    assert logits.shape == (1, BASE["block_size"] * 2, VOCAB)


def test_learned_encoding_refuses_a_context_it_has_no_embeddings_for():
    """Fails loudly and names the fix, rather than indexing out of range."""
    model = build(pos_encoding="learned").eval()
    with pytest.raises(ValueError, match="pos_capacity"):
        model(torch.randint(0, VOCAB, (1, BASE["block_size"] + 1)))


def test_pos_capacity_lets_a_learned_model_run_past_its_training_context():
    """Those extra embeddings are untrained, which is the point of the study."""
    model = build(pos_encoding="learned", pos_capacity=BASE["block_size"] * 2).eval()
    with torch.no_grad():
        logits, _ = model(torch.randint(0, VOCAB, (1, BASE["block_size"] * 2)))
    assert logits.shape == (1, BASE["block_size"] * 2, VOCAB)


# --------------------------------------------------------- model invariants


@pytest.mark.parametrize("pos_encoding,norm_placement", ALL_CONDITIONS)
def test_loss_at_init_is_close_to_a_uniform_prediction(pos_encoding, norm_placement):
    """ln(vocab) is what an untrained model should score. Far off means broken init.

    Targets must be independent of the inputs. Reusing the input as the target
    asks the model to predict the *current* token, which weight tying already
    solves at initialisation: the residual stream carries the token embedding
    and the tied head scores it against itself. That scored 4.26 against a
    ln(128)=4.85 uniform baseline, and most strongly under RoPE, which is the
    one encoding that leaves the residual stream undiluted by a position
    vector.
    """
    model = build(pos_encoding=pos_encoding, norm_placement=norm_placement).eval()
    inputs = torch.randint(0, VOCAB, (4, 12))
    targets = torch.randint(0, VOCAB, (4, 12))
    with torch.no_grad():
        _, loss = model(inputs, targets)
    assert loss.item() == pytest.approx(math.log(VOCAB), abs=0.3)


@pytest.mark.parametrize("pos_encoding,norm_placement", ALL_CONDITIONS)
def test_every_condition_produces_the_expected_logit_shape(pos_encoding, norm_placement):
    model = build(pos_encoding=pos_encoding, norm_placement=norm_placement)
    logits, loss = model(torch.randint(0, VOCAB, (3, 9)))
    assert logits.shape == (3, 9, VOCAB)
    assert loss is None, "loss must be None when no targets are given"


@pytest.mark.parametrize("n_head", [1, 2, 4, 8])
def test_head_count_does_not_change_parameter_count(n_head):
    """Study A3's controlled variable.

    Heads reshape a fixed d_model, so every condition has identical capacity
    and any difference is attributable to attention structure alone. If this
    fails, A3 is confounded and its conclusion is worthless.
    """
    assert build(n_head=n_head).num_params() == build(n_head=BASE["n_head"]).num_params()


def test_norm_placement_does_not_change_parameter_count():
    """Study A1's controlled variable: same parameters, different arrangement."""
    assert build(norm_placement="pre").num_params() == build(norm_placement="post").num_params()


def test_weight_tying_shares_one_matrix():
    model = build(tie_weights=True)
    assert model.lm_head.weight is model.token_embedding.weight


def test_untied_weights_are_independent_and_cost_more_parameters():
    tied, untied = build(tie_weights=True), build(tie_weights=False)
    assert untied.lm_head.weight is not untied.token_embedding.weight
    assert untied.num_params() - tied.num_params() == VOCAB * BASE["d_model"]


def test_sinusoidal_table_is_not_a_trainable_parameter():
    """A fixed encoding must not drift into the optimizer."""
    model = build(pos_encoding="sinusoidal")
    assert not any(p is model.position_table for p in model.parameters())


@pytest.mark.parametrize("pos_encoding,norm_placement", ALL_CONDITIONS)
def test_every_parameter_receives_a_gradient(pos_encoding, norm_placement):
    """A parameter with no gradient is dead weight the optimizer will never move."""
    model = build(pos_encoding=pos_encoding, norm_placement=norm_placement)
    idx = torch.randint(0, VOCAB, (2, 8))
    _, loss = model(idx, idx)
    loss.backward()

    dead = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not dead, f"parameters received no gradient: {dead}"


def test_same_seed_builds_the_same_model():
    a, b = build(), build()
    for (_, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_dropout_is_inactive_in_eval_mode():
    model = build(dropout=0.5).eval()
    idx = torch.randint(0, VOCAB, (2, 8))
    with torch.no_grad():
        first, _ = model(idx)
        second, _ = model(idx)
    assert torch.equal(first, second)


def test_dropout_is_active_in_training_mode():
    model = build(dropout=0.5).train()
    idx = torch.randint(0, VOCAB, (2, 8))
    assert not torch.equal(model(idx)[0], model(idx)[0])


def test_attention_module_rejects_rope_without_rotation_tables():
    """A wiring mistake must fail loudly rather than silently drop position info."""
    attn = CausalSelfAttention(GPTConfig(**{**BASE, "pos_encoding": "rope"}))
    with pytest.raises(RuntimeError, match="rope"):
        attn(torch.randn(1, 8, BASE["d_model"]))


@pytest.mark.parametrize("norm_placement", NORM_PLACEMENTS)
def test_block_preserves_shape(norm_placement):
    block = Block(GPTConfig(**{**BASE, "norm_placement": norm_placement}))
    x = torch.randn(2, 8, BASE["d_model"])
    assert block(x).shape == x.shape


# --------------------------------------------------------------- generation


def test_generate_appends_exactly_the_requested_tokens():
    model = build()
    prompt = torch.randint(0, VOCAB, (2, 4))
    out = model.generate(prompt, max_new_tokens=6)
    assert out.shape == (2, 10)
    assert torch.equal(out[:, :4], prompt), "the prompt must be preserved"


def test_greedy_generation_is_deterministic():
    model = build()
    prompt = torch.randint(0, VOCAB, (1, 4))
    assert torch.equal(
        model.generate(prompt, 8, temperature=0), model.generate(prompt, 8, temperature=0)
    )


def test_top_k_of_one_is_equivalent_to_greedy():
    """Two independent paths to the same answer; a check on the filtering logic."""
    model = build()
    prompt = torch.randint(0, VOCAB, (1, 4))
    assert torch.equal(model.generate(prompt, 8, top_k=1), model.generate(prompt, 8, temperature=0))


def test_generation_crops_context_to_the_training_block_size():
    model = build()
    long_prompt = torch.randint(0, VOCAB, (1, BASE["block_size"] * 2))
    assert model.generate(long_prompt, 3).shape == (1, BASE["block_size"] * 2 + 3)


def test_generated_tokens_are_inside_the_vocabulary():
    model = build()
    out = model.generate(torch.randint(0, VOCAB, (2, 4)), 20, temperature=1.0)
    assert int(out.max()) < VOCAB and int(out.min()) >= 0


def test_generate_rejects_negative_temperature():
    with pytest.raises(ValueError, match="temperature"):
        build().generate(torch.randint(0, VOCAB, (1, 4)), 2, temperature=-1.0)


def test_generate_restores_the_previous_training_mode():
    """Leaving a model in eval mode mid-training would silently disable dropout."""
    model = build().train()
    model.generate(torch.randint(0, VOCAB, (1, 4)), 2)
    assert model.training
