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
