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

from dataclasses import dataclass
from typing import Literal

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
