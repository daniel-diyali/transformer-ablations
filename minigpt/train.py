"""The training loop, written by hand.

Runs are budgeted in **tokens processed**, never in epochs or steps. Two
ablation conditions with different batch shapes must still see exactly the
same amount of data, or the comparison between them means nothing.

    python -m minigpt.train --token-budget 1000000
"""

from __future__ import annotations

import math
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from minigpt.model import GPTConfig


def pick_device(requested: str | None = None) -> str:
    """Choose an accelerator, and say so loudly when falling back to CPU.

    A run that silently lands on CPU looks like a mysteriously slow run three
    hours later, so the fallback announces itself.
    """
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    print("WARNING: no CUDA or MPS device found, training on CPU. This will be very slow.")
    return "cpu"


@dataclass(frozen=True)
class TrainConfig:
    """Everything needed to reproduce one run.

    Serialised into the run directory, so a result can always be traced back
    to the exact settings that produced it.
    """

    model: GPTConfig
    data_dir: Path = Path("data")
    out_dir: Path = Path("runs")
    run_name: str | None = None

    # --- budget ---
    # Tokens, not steps: the unit that makes conditions comparable.
    token_budget: int = 50_000_000
    batch_size: int = 32
    grad_accum: int = 1

    # --- optimizer ---
    lr: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_fraction: float = 0.02
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0

    # --- evaluation ---
    eval_interval_tokens: int = 2_000_000
    eval_batches: int = 40
    eval_block_size: int | None = None

    # --- reproducibility and plumbing ---
    seed: int = 0
    device: str | None = None
    log_wandb: bool = False
    wandb_project: str = "transformer-ablations"

    def __post_init__(self) -> None:
        if self.token_budget < 1:
            raise ValueError(f"token_budget must be positive, got {self.token_budget}")
        if self.batch_size < 1 or self.grad_accum < 1:
            raise ValueError("batch_size and grad_accum must both be at least 1")
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError(f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}")
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0, 1], got {self.min_lr_ratio}")
        if self.eval_block_size is not None and self.eval_block_size < 1:
            raise ValueError("eval_block_size must be positive when set")

    @property
    def tokens_per_step(self) -> int:
        """Data consumed per optimizer step, held constant across conditions."""
        return self.batch_size * self.grad_accum * self.model.block_size

    @property
    def warmup_tokens(self) -> int:
        return int(self.token_budget * self.warmup_fraction)

    @property
    def min_lr(self) -> float:
        return self.lr * self.min_lr_ratio

    def run_dir(self) -> Path:
        if self.run_name:
            return self.out_dir / self.run_name
        model = self.model
        return self.out_dir / f"{model.pos_encoding}-{model.norm_placement}-s{self.seed}"


def provenance(device: str) -> dict:
    """What a future reader needs to explain a number in results.jsonl."""
    return {
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "torch": torch.__version__,
        "device": device,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=5, check=False)
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _git_sha() -> str | None:
    return _git("rev-parse", "HEAD")


def _git_dirty() -> bool | None:
    status = _git("status", "--porcelain")
    return None if status is None else bool(status)


# ------------------------------------------------------ optimizer and schedule


def build_optimizer(model: torch.nn.Module, cfg: TrainConfig) -> torch.optim.AdamW:
    """AdamW with weight decay applied only to matrices.

    Biases and LayerNorm gains are one-dimensional and are excluded. Decaying
    a LayerNorm gain pulls it toward zero, which shrinks the signal the norm
    exists to standardise — a quiet bug that shows up as slightly worse loss
    and nothing else.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.dim() >= 2]
    no_decay = [p for p in trainable if p.dim() < 2]

    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=cfg.betas,
    )


def lr_at_token(tokens_seen: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay, measured in tokens rather than steps.

    Keying the schedule to tokens means two conditions with different batch
    shapes follow the same curve over the same data, instead of one of them
    decaying faster because it happens to take more steps.
    """
    warmup = cfg.warmup_tokens
    if warmup > 0 and tokens_seen < warmup:
        # +1 so the very first step has a non-zero learning rate.
        return cfg.lr * (tokens_seen + 1) / warmup

    span = max(cfg.token_budget - warmup, 1)
    progress = min(max((tokens_seen - warmup) / span, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
