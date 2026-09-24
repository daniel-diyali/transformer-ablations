"""A GPT-style decoder, written to be read.

Attention is computed from primitives rather than delegated to
`nn.MultiheadAttention` or `F.scaled_dot_product_attention`, because showing
the mechanism is the point of the project. The fast path is used only as a
numerical-equivalence target in the tests.

Every dimension under ablation is a field on `GPTConfig` rather than a
separate code path. One model class covers all conditions, so conditions
cannot silently diverge in some unrelated way, and one test suite covers
them all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

NormPlacement = Literal["pre", "post"]
PosEncoding = Literal["learned", "sinusoidal", "rope"]


@dataclass(frozen=True)
class GPTConfig:
    """A complete description of one model.

    Frozen so a config cannot drift mid-run, and hashable so the sweep runner
    can use it as a run identity.
    """

    vocab_size: int
    n_layer: int
    n_head: int
    d_model: int
    block_size: int
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True

    # --- ablation dimensions ---
    norm_placement: NormPlacement = "pre"
    pos_encoding: PosEncoding = "learned"

    # Positions the model can represent. Defaults to block_size. Raising it
    # lets a learned-encoding model run at evaluation contexts longer than it
    # trained on, where those extra embeddings are still at their random
    # initialisation — which is precisely why learned encodings do not
    # extrapolate, and is the effect study A2 measures.
    pos_capacity: int | None = None

    # RoPE rotation base. 10000 is the value from the original paper.
    rope_base: float = 10_000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_head={self.n_head}")
        if self.pos_encoding == "rope" and self.head_dim % 2 != 0:
            # RoPE rotates coordinate pairs, so an odd head dimension has a
            # leftover coordinate with no partner.
            raise ValueError(f"rope needs an even head_dim, got {self.head_dim}")
        for name in ("vocab_size", "n_layer", "n_head", "d_model", "block_size"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.pos_capacity is not None and self.pos_capacity < self.block_size:
            raise ValueError(
                f"pos_capacity={self.pos_capacity} is below block_size={self.block_size}"
            )

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head

    @property
    def n_positions(self) -> int:
        """How many distinct positions the model can encode."""
        return self.pos_capacity if self.pos_capacity is not None else self.block_size


# ------------------------------------------------------- positional encoding
#
# Three schemes, and they do not enter the model at the same place.
#
#   learned / sinusoidal  add a vector to the token embedding, once, before
#                         the first block.
#   rope                  rotates q and k inside every attention layer and
#                         never touches the residual stream.
#
# That asymmetry is inherent to the methods, not an artefact of this code.


def sinusoidal_encoding(n_positions: int, d_model: int, base: float = 10_000.0) -> Tensor:
    """Fixed sin/cos position table from "Attention Is All You Need", section 3.5.

    Even dimensions carry sine, odd carry cosine, with wavelengths in a
    geometric progression. Deterministic, so it extends to any length without
    training.
    """
    position = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    div = torch.exp(-math.log(base) * i / d_model)

    pe = torch.zeros(n_positions, d_model)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)[:, : pe[:, 1::2].shape[1]]
    return pe


def rope_tables(
    seq_len: int,
    head_dim: int,
    base: float = 10_000.0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """cos and sin tables for rotary embedding, each `(seq_len, head_dim)`.

    Computed for whatever length is asked for, which is what lets a RoPE model
    be evaluated beyond its training context at all.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    angles = torch.outer(torch.arange(seq_len, device=device).float(), inv_freq)
    # Duplicated so each half of the head dimension rotates against the other.
    emb = torch.cat([angles, angles], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate `(B, H, T, head_dim)` by position-dependent angles.

    Rotating both q and k makes their dot product depend only on the distance
    between the two positions, never on where the pair sits in the sequence.
    That relative-position property is the whole reason RoPE extrapolates, and
    the tests assert it directly.
    """
    t = x.shape[-2]
    cos, sin = cos[:t].unsqueeze(0).unsqueeze(0), sin[:t].unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------- attention


def causal_attention(q: Tensor, k: Tensor, v: Tensor, dropout: nn.Dropout | None = None) -> Tensor:
    """Scaled dot-product attention with a causal mask, from primitives.

    Shapes are `(B, H, T, head_dim)`. This is deliberately the slow, explicit
    version: score, scale, mask, normalise, average. It is equivalent to
    `F.scaled_dot_product_attention(..., is_causal=True)`, and the tests
    assert that equivalence numerically rather than taking it on faith.

    The mask is built here rather than passed in so the function stays honest
    about what "causal" means: position t may attend to positions <= t, and
    the upper triangle is removed before the softmax so masked positions
    contribute exactly zero probability rather than a small one.
    """
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])

    t_q, t_k = q.shape[-2], k.shape[-2]
    causal = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).tril(diagonal=t_k - t_q)
    scores = scores.masked_fill(~causal, float("-inf"))

    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention. One fused qkv projection, one output projection."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor, cos: Tensor | None = None, sin: Tensor | None = None) -> Tensor:
        b, t, c = x.shape
        n_head, head_dim = self.cfg.n_head, self.cfg.head_dim

        # (B, T, 3C) -> three (B, H, T, head_dim)
        q, k, v = self.qkv(x).split(c, dim=2)
        q, k, v = (tensor.view(b, t, n_head, head_dim).transpose(1, 2) for tensor in (q, k, v))

        if self.cfg.pos_encoding == "rope":
            if cos is None or sin is None:
                raise RuntimeError("rope selected but no rotation tables were supplied")
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        y = causal_attention(q, k, v, self.attn_dropout if self.training else None)
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.resid_dropout(self.proj(y))


# -------------------------------------------------------------------- block


class MLP(nn.Module):
    """Position-wise feed-forward, widened 4x as in the original transformer."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x), approximate="tanh")))


class Block(nn.Module):
    """One transformer layer: attention, then MLP, each around a residual.

    `norm_placement` is study A1's dimension and the only thing that differs
    between its two conditions:

        pre   x = x + attn(norm(x))      the norm sits inside the branch, so
                                         the residual path stays unnormalised
                                         from embedding to output
        post  x = norm(x + attn(x))      the norm sits on the residual path
                                         itself, rescaling it at every layer

    Pre-norm is what every current decoder uses. The point of running the
    comparison is to produce the evidence for that rather than assert it.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.norm_attn = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.norm_mlp = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x: Tensor, cos: Tensor | None = None, sin: Tensor | None = None) -> Tensor:
        if self.cfg.norm_placement == "pre":
            x = x + self.attn(self.norm_attn(x), cos, sin)
            return x + self.mlp(self.norm_mlp(x))
        x = self.norm_attn(x + self.attn(x, cos, sin))
        return self.norm_mlp(x + self.mlp(x))


# -------------------------------------------------------------------- model


class GPT(nn.Module):
    """Decoder-only transformer.

    `forward` returns `(logits, loss)`, where loss is None unless targets are
    supplied — so the same call serves training and generation.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.pos_encoding == "learned":
            self.position_embedding = nn.Embedding(cfg.n_positions, cfg.d_model)
        elif cfg.pos_encoding == "sinusoidal":
            # Not a parameter: fixed, and excluded from the optimizer.
            self.register_buffer(
                "position_table",
                sinusoidal_encoding(cfg.n_positions, cfg.d_model),
                persistent=False,
            )

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm_final = nn.LayerNorm(cfg.d_model, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            # Input and output embeddings share one matrix. Saves vocab*d_model
            # parameters and ties the two views of "what this token means".
            self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)
        # Residual projections are scaled down by depth so the residual stream
        # does not grow as layers are stacked (GPT-2, section 2.3).
        scale = (2 * cfg.n_layer) ** -0.5
        for name, param in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(param, mean=0.0, std=0.02 * scale)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count. Position embeddings are excluded by convention."""
        total = sum(p.numel() for p in self.parameters())
        if non_embedding and self.cfg.pos_encoding == "learned":
            total -= self.position_embedding.weight.numel()
        return total

    def forward(self, idx: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        _, t = idx.shape
        cfg = self.cfg

        x = self.token_embedding(idx)
        cos = sin = None

        if cfg.pos_encoding == "learned":
            if t > cfg.n_positions:
                raise ValueError(
                    f"sequence length {t} exceeds the {cfg.n_positions} learned position "
                    "embeddings this model has. Raise pos_capacity to evaluate beyond the "
                    "training context."
                )
            x = x + self.position_embedding(torch.arange(t, device=idx.device))
        elif cfg.pos_encoding == "sinusoidal":
            # Deterministic, so it extends past n_positions for free.
            table = (
                self.position_table
                if t <= cfg.n_positions
                else sinusoidal_encoding(t, cfg.d_model).to(idx.device)
            )
            x = x + table[:t].to(x.dtype)
        else:
            cos, sin = rope_tables(t, cfg.head_dim, cfg.rope_base, idx.device, x.dtype)

        x = self.drop(x)
        for block in self.blocks:
            x = block(x, cos, sin)
        logits = self.lm_head(self.norm_final(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
