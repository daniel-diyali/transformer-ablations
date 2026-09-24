"""The training loop, written by hand.

Runs are budgeted in **tokens processed**, never in epochs or steps. Two
ablation conditions with different batch shapes must still see exactly the
same amount of data, or the comparison between them means nothing.

    python -m minigpt.train --token-budget 1000000
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from minigpt.data import get_batch, load_meta, load_tokens
from minigpt.model import GPT, GPTConfig

DONE_MARKER = "DONE.json"
FAILED_MARKER = "FAILED.json"


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
        # Coerce paths so a CLI, a JSON round-trip, or a caller passing a
        # string all produce the same config. Frozen dataclass, hence setattr.
        for name in ("data_dir", "out_dir"):
            value = getattr(self, name)
            if not isinstance(value, Path):
                object.__setattr__(self, name, Path(value))

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


# --------------------------------------------------------------- checkpoints

CHECKPOINT = "ckpt.pt"


def _config_to_dict(cfg: TrainConfig) -> dict:
    """Serialise a config, with Paths as strings so it round-trips through JSON."""
    raw = asdict(cfg)
    raw["data_dir"] = str(cfg.data_dir)
    raw["out_dir"] = str(cfg.out_dir)
    return raw


def _save_checkpoint(
    run_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    step: int,
    tokens_seen: int,
    next_eval: int,
    elapsed_s: float,
    batch_generator: torch.Generator,
) -> None:
    """Write atomically, so an interrupt cannot leave a half-written checkpoint.

    RNG state is part of the checkpoint. Without it a resumed run would draw a
    different sequence of batches than an uninterrupted one, and the two would
    stop being comparable — which is exactly what the ablations need them to be.
    """
    tmp = run_dir / (CHECKPOINT + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": _config_to_dict(cfg),
            "step": step,
            "tokens_seen": tokens_seen,
            "next_eval": next_eval,
            "elapsed_s": elapsed_s,
            "batch_generator": batch_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    tmp.replace(run_dir / CHECKPOINT)


def _maybe_resume(
    run_dir: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, cfg: TrainConfig
) -> dict | None:
    """Reload a checkpoint, refusing any that does not match the current config.

    A mismatched resume silently corrupts a comparison: the run would carry one
    condition's weights under another condition's name. Better to stop.
    """
    path = run_dir / CHECKPOINT
    if not path.exists():
        return None

    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["config"] != _config_to_dict(cfg):
        raise RuntimeError(
            f"{path} was written by a different config. Delete the run directory to "
            "start over, or point out_dir somewhere else."
        )

    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    print(f"resuming {run_dir} at {state['tokens_seen']:,} tokens")
    return state


# ------------------------------------------------------------------ evaluation

# Fixed so every evaluation, in every run, scores the same validation batches.
# A moving evaluation set would add noise that looks exactly like a real
# difference between conditions.
EVAL_SEED = 1234


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    tokens: np.ndarray,
    cfg: TrainConfig,
    device: str,
    block_size: int | None = None,
) -> float:
    """Mean loss over a fixed set of validation batches."""
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(EVAL_SEED)
    block = block_size or cfg.eval_block_size or cfg.model.block_size

    total = 0.0
    for _ in range(cfg.eval_batches):
        x, y = get_batch(tokens, cfg.batch_size, block, device, generator)
        _, loss = model(x, y)
        total += loss.item()

    model.train(was_training)
    return total / cfg.eval_batches


def _wandb_run(cfg: TrainConfig, device: str):
    """Optional second logging destination. JSONL remains the source of truth.

    Nothing in the pipeline reads from W&B, so a missing account, a network
    failure or an uninstalled package costs a run nothing but a dashboard.
    """
    if not cfg.log_wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "--wandb requires the optional extra: uv pip install -e '.[wandb]'"
        ) from exc

    return wandb.init(
        project=cfg.wandb_project,
        name=cfg.run_dir().name,
        config={**_config_to_dict(cfg), **provenance(device)},
    )


# -------------------------------------------------------------- training loop


def train(cfg: TrainConfig, resume: bool = True) -> dict:
    """Train one model to its token budget. Returns the run record."""
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / DONE_MARKER).exists():
        print(f"{run_dir} is already complete, skipping")
        return json.loads((run_dir / DONE_MARKER).read_text())

    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)

    train_tokens = load_tokens("train", cfg.data_dir)
    val_tokens = load_tokens("val", cfg.data_dir)

    model = GPT(cfg.model).to(device)
    optimizer = build_optimizer(model, cfg)
    model.train()

    state = _maybe_resume(run_dir, model, optimizer, cfg) if resume else None
    tokens_seen = state["tokens_seen"] if state else 0
    step = state["step"] if state else 0
    next_eval = state["next_eval"] if state else 0

    (run_dir / "config.json").write_text(
        json.dumps({"config": _config_to_dict(cfg), "provenance": provenance(device)}, indent=2)
        + "\n"
    )

    batch_generator = torch.Generator().manual_seed(cfg.seed)
    if state is not None:
        batch_generator.set_state(state["batch_generator"])

    wandb_run = _wandb_run(cfg, device)
    metrics_path = run_dir / "metrics.jsonl"
    started = time.perf_counter()
    elapsed_before = state["elapsed_s"] if state else 0.0

    while tokens_seen < cfg.token_budget:
        lr = lr_at_token(tokens_seen, cfg)
        set_lr(optimizer, lr)
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        for _ in range(cfg.grad_accum):
            x, y = get_batch(
                train_tokens, cfg.batch_size, cfg.model.block_size, device, batch_generator
            )
            _, loss = model(x, y)
            (loss / cfg.grad_accum).backward()
            step_loss += loss.item() / cfg.grad_accum
            tokens_seen += x.numel()

        if not math.isfinite(step_loss):
            # A diverged run is data, not an error to hide. Post-norm may
            # legitimately blow up, and that is the finding.
            record = {
                "status": "failed",
                "reason": "non-finite loss",
                "step": step,
                "tokens_seen": tokens_seen,
                "last_loss": step_loss,
            }
            (run_dir / FAILED_MARKER).write_text(json.dumps(record, indent=2) + "\n")
            print(f"ABORT: non-finite loss at step {step}, {tokens_seen:,} tokens")
            return record

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        step += 1

        if tokens_seen >= next_eval or tokens_seen >= cfg.token_budget:
            elapsed = elapsed_before + time.perf_counter() - started
            entry = {
                "step": step,
                "tokens": tokens_seen,
                "train_loss": round(step_loss, 6),
                "val_loss": round(evaluate(model, val_tokens, cfg, device), 6),
                "lr": lr,
                "elapsed_s": round(elapsed, 2),
                "tokens_per_s": round(tokens_seen / max(elapsed, 1e-9)),
            }
            with metrics_path.open("a") as fh:
                fh.write(json.dumps(entry) + "\n")
            if wandb_run is not None:
                wandb_run.log(entry, step=entry["step"])
            print(
                f"step {entry['step']:>6} | {entry['tokens']:>12,} tok | "
                f"train {entry['train_loss']:.4f} | val {entry['val_loss']:.4f} | "
                f"lr {lr:.2e} | {entry['tokens_per_s']:,} tok/s"
            )
            next_eval = tokens_seen + cfg.eval_interval_tokens
            _save_checkpoint(
                run_dir,
                model,
                optimizer,
                cfg,
                step,
                tokens_seen,
                next_eval,
                elapsed,
                batch_generator,
            )

    elapsed = elapsed_before + time.perf_counter() - started
    record = {
        "status": "completed",
        "step": step,
        "tokens_seen": tokens_seen,
        "final_val_loss": evaluate(model, val_tokens, cfg, device),
        "best_val_loss": _best_val_loss(metrics_path),
        "elapsed_s": round(elapsed, 2),
        "tokens_per_s": round(tokens_seen / max(elapsed, 1e-9)),
        "device": device,
        "run_dir": str(run_dir),
    }
    (run_dir / DONE_MARKER).write_text(json.dumps(record, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.summary.update(record)
        wandb_run.finish()
    return record


def _best_val_loss(metrics_path: Path) -> float | None:
    if not metrics_path.exists():
        return None
    losses = [
        json.loads(line)["val_loss"] for line in metrics_path.read_text().splitlines() if line
    ]
    return min(losses) if losses else None


# ---------------------------------------------------------------------- cli


def build_config(args: argparse.Namespace) -> TrainConfig:
    """Assemble a config from CLI arguments.

    vocab_size is read from the corpus metadata rather than accepted as a
    flag, so a model can never be built with a vocabulary its data does not
    match — a mismatch that would train happily and produce nonsense.
    """
    meta = load_meta(Path(args.data_dir))
    model = GPTConfig(
        vocab_size=meta["vocab_size"],
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        block_size=args.block_size,
        dropout=args.dropout,
        pos_encoding=args.pos_encoding,
        norm_placement=args.norm_placement,
        pos_capacity=args.pos_capacity,
    )
    return TrainConfig(
        model=model,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        run_name=args.run_name,
        token_budget=args.token_budget,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        warmup_fraction=args.warmup_fraction,
        eval_interval_tokens=args.eval_interval_tokens,
        eval_batches=args.eval_batches,
        seed=args.seed,
        device=args.device,
        log_wandb=args.wandb,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)

    model_args = parser.add_argument_group("model")
    model_args.add_argument("--n-layer", type=int, default=6)
    model_args.add_argument("--n-head", type=int, default=6)
    model_args.add_argument("--d-model", type=int, default=384)
    model_args.add_argument("--block-size", type=int, default=256)
    model_args.add_argument("--dropout", type=float, default=0.0)
    model_args.add_argument(
        "--pos-encoding", choices=["learned", "sinusoidal", "rope"], default="learned"
    )
    model_args.add_argument("--norm-placement", choices=["pre", "post"], default="pre")
    model_args.add_argument(
        "--pos-capacity", type=int, default=None, help="positions to allocate; see study A2"
    )

    train_args = parser.add_argument_group("training")
    train_args.add_argument("--token-budget", type=int, default=50_000_000)
    train_args.add_argument("--batch-size", type=int, default=32)
    train_args.add_argument("--grad-accum", type=int, default=1)
    train_args.add_argument("--lr", type=float, default=6e-4)
    train_args.add_argument("--warmup-fraction", type=float, default=0.02)
    train_args.add_argument("--eval-interval-tokens", type=int, default=2_000_000)
    train_args.add_argument("--eval-batches", type=int, default=40)
    train_args.add_argument("--seed", type=int, default=0)

    run_args = parser.add_argument_group("run")
    run_args.add_argument("--data-dir", type=Path, default=Path("data"))
    run_args.add_argument("--out-dir", type=Path, default=Path("runs"))
    run_args.add_argument("--run-name", default=None)
    run_args.add_argument("--device", default=None)
    run_args.add_argument("--wandb", action="store_true", help="optional; JSONL stays canonical")
    run_args.add_argument(
        "--no-resume", action="store_true", help="ignore any checkpoint in the run directory"
    )

    args = parser.parse_args(argv)
    cfg = build_config(args)

    print(f"run:    {cfg.run_dir()}")
    print(f"model:  {cfg.model.n_layer}L/{cfg.model.n_head}H/{cfg.model.d_model}d, ", end="")
    print(f"{cfg.model.pos_encoding}, {cfg.model.norm_placement}-norm")
    print(f"budget: {cfg.token_budget:,} tokens, {cfg.tokens_per_step:,} per step")

    record = train(cfg, resume=not args.no_resume)
    print(json.dumps(record, indent=2))
    return 0 if record.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
