"""Run records to figures.

Pure post-processing: reads the JSONL a run wrote and emits charts. Imports
no torch and knows nothing about how a run was produced, so every figure can
be regenerated without retraining anything.

    python -m minigpt.analysis loss-curve --run runs/baseline-50M
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib

# Non-interactive backend: figures are written to disk, never displayed, so
# this works the same over SSH and in CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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


def plot_loss_curve(
    run_dir: Path,
    out_path: Path | None = None,
    title: str | None = None,
    log_y: bool = True,
) -> Path:
    """Training and validation loss against tokens seen.

    Both series share one axis because the question a reader has is whether
    they diverge, and that is only answerable when they are drawn together.

    The y-axis is logarithmic by default. Loss falls from ~9 to under 2, and
    on a linear axis that first plunge swallows the vertical space, leaving
    the plateau — where the differences between conditions actually live —
    squashed into a few pixels.
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
    ax.grid(True, alpha=0.2, linewidth=0.6, which="both")
    ax.spines[["top", "right"]].set_visible(False)

    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())
        ax.tick_params(axis="y", which="minor", labelsize=7)
    else:
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
    parser.add_argument(
        "command", choices=["loss-curve", "summary", "study", "context-scaling", "table"]
    )
    parser.add_argument("--run", type=Path, default=None, help="a single run directory")
    parser.add_argument("--sweep", type=Path, default=Path("runs"), help="sweep output directory")
    parser.add_argument("--study", default=None, help="study name for `study`")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--linear-y", action="store_true", help="linear loss axis")
    args = parser.parse_args(argv)

    if args.command in ("loss-curve", "summary"):
        if args.run is None:
            raise SystemExit(f"--run is required for {args.command}")
        if args.command == "loss-curve":
            print(plot_loss_curve(args.run, args.out, args.title, log_y=not args.linear_y))
        else:
            print(json.dumps(summarize(args.run), indent=2))
        return 0

    if args.command == "study":
        if args.study is None:
            raise SystemExit("--study is required for `study`")
        print(plot_study(load_sweep(args.sweep), args.study, args.out, args.title))
        return 0

    if args.command == "context-scaling":
        print(plot_context_scaling(load_context_eval(args.sweep), args.out, args.title))
        return 0

    records = load_sweep(args.sweep)
    studies = [args.study] if args.study else sorted({r["study"] for r in records})
    for study in studies:
        print(f"\n{study}")
        for row in study_summary(records, study):
            seeds = ", ".join(f"{loss:.4f}" for loss in row["losses"])
            print(
                f"  {row['condition']:<12} mean {row['mean']:.4f}  "
                f"spread {row['spread']:.4f}  n={row['n_seeds']}  [{seeds}]"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ------------------------------------------------------------ sweep results

CONDITION_COLOURS = ["#1f4e8c", "#c2571a", "#2a7f62", "#7a3b8f", "#8c1f3d"]


def load_sweep(out_dir: Path) -> list[dict]:
    """Every run record the sweep wrote."""
    path = out_dir / "results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; no sweep has been run in {out_dir}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def study_summary(records: list[dict], study: str) -> list[dict]:
    """Per-condition statistics across seeds, in condition order.

    Reports spread as well as mean. With three seeds a mean alone cannot tell
    you whether a gap between conditions is real, and reporting it alone is
    the mistake most public ablation repos make.
    """
    rows = [r for r in records if r["study"] == study and r["status"] == "completed"]
    if not rows:
        raise ValueError(f"no completed runs for study {study!r}")

    summary = []
    for condition in dict.fromkeys(r["condition"] for r in rows):
        losses = sorted(r["final_val_loss"] for r in rows if r["condition"] == condition)
        summary.append(
            {
                "condition": condition,
                "n_seeds": len(losses),
                "mean": statistics.fmean(losses),
                "min": losses[0],
                "max": losses[-1],
                "spread": losses[-1] - losses[0],
                "stdev": statistics.stdev(losses) if len(losses) > 1 else 0.0,
                "losses": losses,
            }
        )
    return summary


def plot_study(
    records: list[dict], study: str, out_path: Path | None = None, title: str | None = None
) -> Path:
    """Final validation loss per condition, with every seed drawn.

    Individual seeds are plotted, not just their mean. If the seeds overlap
    between conditions, the chart has to show that the difference is inside
    the noise — which is the whole point of running three of them.
    """
    summary = study_summary(records, study)

    fig, ax = plt.subplots(figsize=(1.6 * len(summary) + 3.0, 4.5), dpi=150)
    for index, row in enumerate(summary):
        colour = CONDITION_COLOURS[index % len(CONDITION_COLOURS)]
        seed_label = "individual seeds" if index == 0 else None
        mean_label = "mean" if index == 0 else None
        ax.scatter(
            [index] * len(row["losses"]),
            row["losses"],
            s=46,
            color=colour,
            alpha=0.75,
            zorder=3,
            label=seed_label,
        )
        ax.hlines(
            row["mean"],
            index - 0.22,
            index + 0.22,
            color=colour,
            linewidth=2.4,
            zorder=4,
            label=mean_label,
        )
        ax.annotate(
            f"{row['mean']:.3f}",
            xy=(index + 0.26, row["mean"]),
            va="center",
            fontsize=9,
            color=colour,
            fontweight="bold",
        )

    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels([row["condition"] for row in summary])
    ax.set_xlim(-0.5, len(summary) - 0.1)
    ax.set_ylabel("final validation loss")
    ax.set_title(
        title or f"{study} — {summary[0]['n_seeds']} seeds per condition", loc="left", fontsize=11
    )
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best", fontsize=9)

    out_path = out_path or FIGURES / f"{study}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def load_context_eval(out_dir: Path) -> list[dict]:
    path = out_dir / "context_eval.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. Run: python -m minigpt.experiments context-eval")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def plot_context_scaling(
    rows: list[dict], out_path: Path | None = None, title: str | None = None
) -> Path:
    """Study A2's headline: validation loss against evaluation context length.

    The training context is marked, because the whole question is what happens
    to the right of that line. Conditions that cannot represent a context are
    annotated rather than silently absent — a missing point reads as lost
    data, whereas "cannot represent" is the finding.
    """
    contexts = sorted({row["context"] for row in rows})
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    train_context = rows[0]["train_context"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)

    for index, condition in enumerate(conditions):
        colour = CONDITION_COLOURS[index % len(CONDITION_COLOURS)]
        means, points = [], []
        for context in contexts:
            losses = [
                r["val_loss"]
                for r in rows
                if r["condition"] == condition
                and r["context"] == context
                and r["val_loss"] is not None
            ]
            if losses:
                means.append((context, statistics.fmean(losses)))
                points.extend((context, loss) for loss in losses)

        if points:
            ax.scatter(*zip(*points, strict=True), s=18, color=colour, alpha=0.45, zorder=3)
        if means:
            ax.plot(
                *zip(*means, strict=True),
                color=colour,
                linewidth=2.0,
                marker="o",
                markersize=4,
                label=condition,
                zorder=4,
            )

        missing = sorted(
            {r["context"] for r in rows if r["condition"] == condition and r["val_loss"] is None}
        )
        if missing:
            ax.annotate(
                f"{condition}: cannot represent\ncontexts beyond {min(missing) // 2}",
                xy=(min(missing), ax.get_ylim()[1]),
                fontsize=8,
                color=colour,
                ha="center",
                va="top",
            )

    ax.axvline(train_context, color="#555555", linestyle="--", linewidth=1.1, zorder=2)
    ax.annotate(
        f"trained at {train_context}",
        xy=(train_context, ax.get_ylim()[1]),
        xytext=(4, -6),
        textcoords="offset points",
        fontsize=8,
        color="#555555",
        va="top",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("evaluation context length (tokens)")
    ax.set_ylabel("validation loss")
    ax.set_title(
        title or "Positional encoding beyond the training context", loc="left", fontsize=11
    )
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, title="encoding", fontsize=9, title_fontsize=9)

    out_path = out_path or FIGURES / "pos_encoding-context-scaling.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
