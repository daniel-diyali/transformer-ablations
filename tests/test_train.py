"""Training loop specification.

The gate is `test_model_can_memorize_a_single_batch`: if the loop cannot
drive loss to nearly zero on fixed data, something in the forward, backward,
optimizer or schedule is wrong, and every later result is meaningless.

A synthetic corpus is written to a temp directory, so nothing here touches
the network or the real 890 MB corpus.
"""

import json
import math
from itertools import pairwise

import numpy as np
import pytest
import torch

from minigpt.model import GPT, GPTConfig
from minigpt.train import (
    DONE_MARKER,
    FAILED_MARKER,
    TrainConfig,
    build_optimizer,
    evaluate,
    lr_at_token,
    provenance,
    train,
)

VOCAB = 64
TINY_MODEL = dict(vocab_size=VOCAB, n_layer=2, n_head=2, d_model=32, block_size=16, dropout=0.0)


@pytest.fixture
def corpus(tmp_path):
    """A small repeating token stream, plus the metadata the loader checks."""
    rng = np.random.default_rng(0)
    train_tokens = rng.integers(0, VOCAB, size=20_000, dtype=np.uint16)
    val_tokens = rng.integers(0, VOCAB, size=4_000, dtype=np.uint16)

    train_tokens.tofile(tmp_path / "train.bin")
    val_tokens.tofile(tmp_path / "val.bin")
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "vocab_size": VOCAB,
                "n_train_tokens": int(train_tokens.size),
                "n_val_tokens": int(val_tokens.size),
            }
        )
    )
    return tmp_path


def make_config(corpus, tmp_path, **overrides) -> TrainConfig:
    defaults = dict(
        model=GPTConfig(**TINY_MODEL),
        data_dir=corpus,
        out_dir=tmp_path / "runs",
        token_budget=8_192,
        batch_size=4,
        eval_interval_tokens=4_096,
        eval_batches=2,
        device="cpu",
        seed=0,
    )
    return TrainConfig(**{**defaults, **overrides})


# ------------------------------------------------------------------ the gate


def test_model_can_memorize_a_single_batch():
    """The sanity check that catches almost every plumbing bug at once.

    A correct forward, backward, optimizer and parameter grouping can drive
    loss on fixed data to nearly zero. A wrong one cannot, no matter how
    plausible the loss curve looks on real data.
    """
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY_MODEL))
    cfg = TrainConfig(model=model.cfg, lr=3e-3, token_budget=1)
    optimizer = build_optimizer(model, cfg)

    x = torch.randint(0, VOCAB, (4, 16))
    y = torch.randint(0, VOCAB, (4, 16))

    initial = None
    for _ in range(300):
        _, loss = model(x, y)
        if initial is None:
            initial = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    assert initial == pytest.approx(math.log(VOCAB), abs=0.5)
    assert loss.item() < 0.1, f"loss stalled at {loss.item():.4f}; the loop is not learning"


# ---------------------------------------------------------------- schedule


def test_warmup_reaches_exactly_the_peak_rate():
    cfg = TrainConfig(model=GPTConfig(**TINY_MODEL), token_budget=1_000_000, lr=1e-3)
    assert lr_at_token(cfg.warmup_tokens - 1, cfg) == pytest.approx(1e-3)


def test_schedule_ends_exactly_at_the_minimum_rate():
    cfg = TrainConfig(model=GPTConfig(**TINY_MODEL), token_budget=1_000_000, lr=1e-3)
    assert lr_at_token(cfg.token_budget, cfg) == pytest.approx(cfg.min_lr)


def test_schedule_never_exceeds_the_peak_rate():
    cfg = TrainConfig(model=GPTConfig(**TINY_MODEL), token_budget=100_000, lr=1e-3)
    assert max(lr_at_token(t, cfg) for t in range(0, 100_001, 500)) <= 1e-3 + 1e-12


def test_schedule_decreases_monotonically_after_warmup():
    cfg = TrainConfig(model=GPTConfig(**TINY_MODEL), token_budget=100_000, lr=1e-3)
    after = [lr_at_token(t, cfg) for t in range(cfg.warmup_tokens, 100_001, 1_000)]
    assert all(b <= a + 1e-12 for a, b in pairwise(after))


def test_schedule_stays_flat_when_warmup_is_disabled():
    cfg = TrainConfig(model=GPTConfig(**TINY_MODEL), token_budget=1_000, warmup_fraction=0.0)
    assert lr_at_token(0, cfg) == pytest.approx(cfg.lr)


# --------------------------------------------------------- optimizer groups


def test_weight_decay_applies_only_to_matrices():
    """Decaying a LayerNorm gain shrinks the signal the norm exists to standardise."""
    model = GPT(GPTConfig(**TINY_MODEL))
    cfg = TrainConfig(model=model.cfg, weight_decay=0.1)
    decay, no_decay = build_optimizer(model, cfg).param_groups

    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    assert {p.dim() for p in decay["params"]} == {2}
    assert {p.dim() for p in no_decay["params"]} == {1}


def test_every_trainable_parameter_lands_in_exactly_one_group():
    model = GPT(GPTConfig(**TINY_MODEL))
    groups = build_optimizer(model, TrainConfig(model=model.cfg)).param_groups
    grouped = sum(len(g["params"]) for g in groups)
    assert grouped == len([p for p in model.parameters() if p.requires_grad])


# ------------------------------------------------------------------- config


def test_tokens_per_step_accounts_for_accumulation():
    cfg = TrainConfig(model=GPTConfig(**TINY_MODEL), batch_size=4, grad_accum=3)
    assert cfg.tokens_per_step == 4 * 3 * 16


def test_string_paths_are_coerced():
    cfg = TrainConfig(model=GPTConfig(**TINY_MODEL), data_dir="data", out_dir="runs")
    assert cfg.data_dir.name == "data" and cfg.out_dir.name == "runs"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"token_budget": 0},
        {"batch_size": 0},
        {"grad_accum": 0},
        {"warmup_fraction": 1.0},
        {"min_lr_ratio": 2.0},
    ],
)
def test_invalid_configs_are_rejected_at_construction(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(model=GPTConfig(**TINY_MODEL), **kwargs)


def test_provenance_records_the_commit_and_whether_the_tree_was_dirty():
    """A result from uncommitted code is not reproducible; record that honestly."""
    record = provenance("cpu")
    assert record["device"] == "cpu"
    assert record["torch"] == torch.__version__
    assert "git_dirty" in record


# --------------------------------------------------------------- evaluation


def test_evaluation_scores_the_same_batches_every_time(corpus, tmp_path):
    """A moving eval set adds noise that looks exactly like a real difference."""
    cfg = make_config(corpus, tmp_path)
    model = GPT(cfg.model)
    tokens = np.fromfile(corpus / "val.bin", dtype=np.uint16)
    assert evaluate(model, tokens, cfg, "cpu") == evaluate(model, tokens, cfg, "cpu")


def test_evaluation_restores_training_mode(corpus, tmp_path):
    cfg = make_config(corpus, tmp_path)
    model = GPT(cfg.model).train()
    evaluate(model, np.fromfile(corpus / "val.bin", dtype=np.uint16), cfg, "cpu")
    assert model.training


# ----------------------------------------------------------- the full loop


def test_a_run_completes_and_records_its_result(corpus, tmp_path):
    record = train(make_config(corpus, tmp_path))
    assert record["status"] == "completed"
    assert record["tokens_seen"] >= 8_192
    assert (tmp_path / "runs" / record["run_dir"].split("/")[-1] / DONE_MARKER).exists()


def test_a_run_writes_metrics_config_and_a_checkpoint(corpus, tmp_path):
    cfg = make_config(corpus, tmp_path)
    train(cfg)
    run_dir = cfg.run_dir()

    for name in ("metrics.jsonl", "config.json", "ckpt.pt", DONE_MARKER):
        assert (run_dir / name).exists(), f"{name} missing"

    entries = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert entries, "no metrics were logged"
    assert {"step", "tokens", "train_loss", "val_loss", "lr", "tokens_per_s"} <= set(entries[0])


def test_the_same_seed_produces_the_same_result(corpus, tmp_path):
    """Two conditions must differ only in what is under study, never in luck."""
    first = train(make_config(corpus, tmp_path, run_name="a"))
    second = train(make_config(corpus, tmp_path, run_name="b"))
    assert first["final_val_loss"] == pytest.approx(second["final_val_loss"], abs=1e-6)


def test_different_seeds_produce_different_results(corpus, tmp_path):
    """Seed variance is real, which is why every condition runs three of them."""
    first = train(make_config(corpus, tmp_path, run_name="a", seed=0))
    second = train(make_config(corpus, tmp_path, run_name="b", seed=1))
    assert first["final_val_loss"] != second["final_val_loss"]


def test_a_completed_run_is_not_repeated(corpus, tmp_path):
    """What makes an interrupted overnight sweep resumable without bookkeeping."""
    cfg = make_config(corpus, tmp_path)
    first = train(cfg)
    mtime = (cfg.run_dir() / "ckpt.pt").stat().st_mtime

    again = train(cfg)
    assert again == first
    assert (cfg.run_dir() / "ckpt.pt").stat().st_mtime == mtime, "the run was redone"


def test_resuming_picks_up_from_the_checkpoint_rather_than_restarting(corpus, tmp_path):
    """An interrupted run continues where it stopped, on identical settings."""
    cfg = make_config(corpus, tmp_path, run_name="r")
    first = train(cfg)

    # Simulate a run that reached its budget but was killed before finishing.
    (cfg.run_dir() / DONE_MARKER).unlink()
    resumed = train(cfg)

    assert resumed["status"] == "completed"
    assert resumed["tokens_seen"] == first["tokens_seen"], "resume restarted from zero"
    assert resumed["step"] == first["step"]


def test_resuming_with_a_changed_token_budget_is_refused(corpus, tmp_path):
    """Intentional: the LR schedule is keyed to the budget.

    Resuming a 4k-token run under an 8k-token budget would follow a curve
    matching neither, producing a result that is not comparable to anything.
    The budget is part of the config, so the config check catches it.
    """
    cfg = make_config(corpus, tmp_path, run_name="r", token_budget=4_096)
    train(cfg)
    (cfg.run_dir() / DONE_MARKER).unlink()

    extended = make_config(corpus, tmp_path, run_name="r", token_budget=8_192)
    with pytest.raises(RuntimeError, match="different config"):
        train(extended)


def test_resuming_with_a_different_config_is_refused(corpus, tmp_path):
    """A mismatched resume would train one condition under another's name."""
    cfg = make_config(corpus, tmp_path, run_name="r")
    train(cfg)
    (cfg.run_dir() / DONE_MARKER).unlink()

    changed = make_config(
        corpus, tmp_path, run_name="r", model=GPTConfig(**{**TINY_MODEL, "n_layer": 3})
    )
    with pytest.raises(RuntimeError, match="different config"):
        train(changed)


def test_a_diverging_run_aborts_and_records_the_failure(corpus, tmp_path):
    """A diverged run is data. Post-norm may genuinely blow up; that is the finding."""
    cfg = make_config(corpus, tmp_path, run_name="boom", lr=1e9, warmup_fraction=0.0)
    record = train(cfg)

    assert record["status"] == "failed"
    assert record["reason"] == "non-finite loss"
    failure = json.loads((cfg.run_dir() / FAILED_MARKER).read_text())
    assert failure["tokens_seen"] > 0, "the failure record must say how far it got"


def test_config_json_captures_provenance(corpus, tmp_path):
    cfg = make_config(corpus, tmp_path)
    train(cfg)
    saved = json.loads((cfg.run_dir() / "config.json").read_text())
    assert saved["config"]["token_budget"] == cfg.token_budget
    assert "git_sha" in saved["provenance"]
