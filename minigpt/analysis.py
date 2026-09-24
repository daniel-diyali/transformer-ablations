"""Run records to figures.

Pure post-processing: reads the JSONL a run wrote and emits charts. Imports
no torch and knows nothing about how a run was produced, so every figure can
be regenerated without retraining anything.

    python -m minigpt.analysis loss-curve --run runs/baseline-50M
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# Non-interactive backend: figures are written to disk, never displayed, so
# this works the same over SSH and in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

FIGURES = Path("figures")

TRAIN_COLOUR = "#7a7a7a"
VAL_COLOUR = "#1f4e8c"


def load_metrics(run_dir: Path) -> list[dict]:
    """Read a run's metrics.jsonl in logged order."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; {run_dir} has no logged metrics")

    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"{path} is empty")
    return entries


def load_record(run_dir: Path) -> dict | None:
    """The run's DONE record, or None if it never finished."""
    done = run_dir / "DONE.json"
    return json.loads(done.read_text()) if done.exists() else None


def plot_loss_curve(run_dir: Path, out_path: Path | None = None, title: str | None = None) -> Path:
    """Training and validation loss against tokens seen.

    Both series share one axis because the question a reader has is whether
    they diverge, and that is only answerable when they are drawn together.
    """
    entries = load_metrics(run_dir)
    tokens = [e["tokens"] / 1e6 for e in entries]
    train = [e["train_loss"] for e in entries]
    val = [e["val_loss"] for e in entries]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.plot(tokens, train, color=TRAIN_COLOUR, linewidth=1.2, label="train", alpha=0.85)
    ax.plot(tokens, val, color=VAL_COLOUR, linewidth=1.8, label="validation")

    # The final validation loss is the number a reader came for.
    ax.annotate(
        f"{val[-1]:.3f}",
        xy=(tokens[-1], val[-1]),
        xytext=(-6, 10),
        textcoords="offset points",
        ha="right",
        color=VAL_COLOUR,
        fontweight="bold",
    )

    ax.set_xlabel("tokens seen (millions)")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(title or f"{run_dir.name} — loss", loc="left", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(bottom=min(min(train), min(val)) * 0.97)

    out_path = out_path or FIGURES / f"{run_dir.name}-loss.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def summarize(run_dir: Path) -> dict:
    """The handful of numbers worth quoting about a run."""
    entries = load_metrics(run_dir)
    record = load_record(run_dir)
    val_losses = [e["val_loss"] for e in entries]

    return {
        "run": run_dir.name,
        "status": (record or {}).get("status", "unfinished"),
        "tokens": entries[-1]["tokens"],
        "steps": entries[-1]["step"],
        "first_val_loss": val_losses[0],
        "final_val_loss": val_losses[-1],
        "best_val_loss": min(val_losses),
        "tokens_per_s": entries[-1]["tokens_per_s"],
        "elapsed_min": round(entries[-1]["elapsed_s"] / 60, 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["loss-curve", "summary"])
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)

    if args.command == "loss-curve":
        print(plot_loss_curve(args.run, args.out, args.title))
    else:
        print(json.dumps(summarize(args.run), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
