"""Ablation studies: declare conditions, expand over seeds, run, aggregate.

A study says what it is testing and what it expects, then names the single
field that differs between its conditions. Everything else is inherited from
one shared base config, so two conditions cannot drift apart in some way
nobody intended.

    python -m minigpt.experiments run --study norm_placement
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path

from minigpt.data import load_meta
from minigpt.model import GPTConfig
from minigpt.train import TrainConfig, _config_to_dict, train

MODEL_FIELDS = {f.name for f in dataclass_fields(GPTConfig)}
TRAIN_FIELDS = {f.name for f in dataclass_fields(TrainConfig)} - {"model"}


def apply_overrides(base: TrainConfig, overrides: dict) -> TrainConfig:
    """Return `base` with `overrides` applied, routed to model or train config.

    An unknown key raises rather than being ignored. A silently dropped
    override is the worst failure this file could have: the sweep would run,
    every condition would be identical, and the resulting chart would look
    like a real finding of "no difference".
    """
    model_changes = {k: v for k, v in overrides.items() if k in MODEL_FIELDS}
    train_changes = {k: v for k, v in overrides.items() if k in TRAIN_FIELDS}

    unknown = set(overrides) - MODEL_FIELDS - TRAIN_FIELDS
    if unknown:
        raise ValueError(
            f"unknown override(s): {sorted(unknown)}. "
            f"Valid model fields: {sorted(MODEL_FIELDS)}. "
            f"Valid training fields: {sorted(TRAIN_FIELDS)}."
        )

    config = base
    if model_changes:
        config = replace(config, model=replace(config.model, **model_changes))
    if train_changes:
        config = replace(config, **train_changes)
    return config


def config_hash(cfg: TrainConfig) -> str:
    """A short, stable fingerprint of everything that defines a run."""
    payload = json.dumps(_config_to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class RunSpec:
    """One cell of a study: a condition, a seed, and the config they produce."""

    study: str
    condition: str
    seed: int
    config: TrainConfig

    @property
    def name(self) -> str:
        return f"{self.study}-{self.condition}-s{self.seed}"

    @property
    def fingerprint(self) -> str:
        return config_hash(self.config)


@dataclass(frozen=True)
class Study:
    """One ablation: a question, a prediction, and the conditions that test it.

    The hypothesis is recorded before anything runs. Writing down what you
    expect and then reporting whether it held is the difference between an
    experiment and a search for a flattering number.
    """

    name: str
    hypothesis: str
    conditions: dict[str, dict]
    seeds: tuple[int, ...] = (0, 1, 2)
    # Applied to every condition — for settings a study needs but is not varying.
    shared: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.conditions) < 2:
            raise ValueError(f"study {self.name!r} needs at least two conditions to compare")
        if not self.seeds:
            raise ValueError(f"study {self.name!r} needs at least one seed")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError(f"study {self.name!r} has duplicate seeds: {self.seeds}")

    def expand(self, base: TrainConfig) -> Iterator[RunSpec]:
        """Every (condition, seed) pair, as a fully resolved run."""
        for condition, overrides in self.conditions.items():
            for seed in self.seeds:
                config = apply_overrides(base, {**self.shared, **overrides})
                config = replace(config, seed=seed, run_name=f"{self.name}-{condition}-s{seed}")
                yield RunSpec(self.name, condition, seed, config)

    def __len__(self) -> int:
        return len(self.conditions) * len(self.seeds)


# ---------------------------------------------------------------- the sweep

RESULTS = "results.jsonl"


def load_results(out_dir: Path) -> dict[str, dict]:
    """Completed run records, keyed by run name."""
    path = out_dir / RESULTS
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {record["run"]: record for record in records}


def append_result(out_dir: Path, record: dict) -> None:
    """Append one record. Written as it happens, so an interrupted sweep
    still leaves every finished run on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / RESULTS).open("a") as fh:
        fh.write(json.dumps(record) + "\n")


def _record_for(spec: RunSpec, outcome: dict) -> dict:
    return {
        "study": spec.study,
        "condition": spec.condition,
        "seed": spec.seed,
        "run": spec.name,
        "fingerprint": spec.fingerprint,
        "status": outcome.get("status", "unknown"),
        "final_val_loss": outcome.get("final_val_loss"),
        "best_val_loss": outcome.get("best_val_loss"),
        "tokens_seen": outcome.get("tokens_seen"),
        "steps": outcome.get("step"),
        "elapsed_s": outcome.get("elapsed_s"),
        "tokens_per_s": outcome.get("tokens_per_s"),
        "reason": outcome.get("reason"),
        "run_dir": str(spec.config.run_dir()),
    }


def run_sweep(studies: Sequence[Study], base: TrainConfig, resume: bool = True) -> list[dict]:
    """Run every condition and seed in `studies`, one at a time.

    A failing run is recorded and the sweep continues. One bad condition must
    not cost a night of compute, and a crash that stops everything is how a
    sweep silently becomes a partial sweep nobody notices.
    """
    out_dir = base.out_dir
    done = load_results(out_dir) if resume else {}
    specs = [spec for study in studies for spec in study.expand(base)]

    print(f"sweep: {len(specs)} runs across {len(studies)} studies")
    records: list[dict] = []

    for index, spec in enumerate(specs, start=1):
        previous = done.get(spec.name)
        if previous is not None:
            if previous.get("fingerprint") != spec.fingerprint:
                raise RuntimeError(
                    f"{spec.name} was already run with a different config "
                    f"({previous.get('fingerprint')} vs {spec.fingerprint}). "
                    "Rename the study or clear its results before rerunning, or the "
                    "sweep would mix settings under one name."
                )
            print(f"[{index}/{len(specs)}] {spec.name}: already done, skipping")
            records.append(previous)
            continue

        print(f"[{index}/{len(specs)}] {spec.name}")
        try:
            outcome = train(spec.config)
        except Exception as exc:
            # Deliberately broad. A failed run is data about that condition,
            # not a reason to abandon the other twenty-six.
            outcome = {"status": "error", "reason": f"{type(exc).__name__}: {exc}"}
            print(f"  ERROR: {outcome['reason']}")

        record = _record_for(spec, outcome)
        append_result(out_dir, record)
        records.append(record)

    _report(records)
    return records


def _report(records: Sequence[dict]) -> None:
    """A short account of what happened, including what did not work."""
    by_status: dict[str, int] = {}
    for record in records:
        by_status[record["status"]] = by_status.get(record["status"], 0) + 1

    tally = ", ".join(f"{n} {status}" for status, n in sorted(by_status.items()))
    print(f"\nsweep finished: {tally}")

    problems = [r for r in records if r["status"] != "completed"]
    if problems:
        print("runs that did not complete:")
        for record in problems:
            print(f"  {record['run']}: {record['status']} — {record.get('reason')}")


# -------------------------------------------------------------- the studies
#
# Hypotheses are written here, before any of these run. Whether each held is
# reported in FINDINGS.md, including the ones that did not.

# Evaluation contexts beyond the training length that study A2 needs. The
# learned condition must allocate embeddings for them up front; those stay at
# their random initialisation because training never reaches that far, which
# is precisely the effect being measured.
A2_MAX_EVAL_CONTEXT = 1024


def sweep_base(vocab_size: int, **overrides) -> TrainConfig:
    """The shared configuration every condition inherits.

    Measured at 13.89M parameters and roughly 27 minutes per 50M-token run.
    """
    model = GPTConfig(vocab_size=vocab_size, n_layer=6, n_head=6, d_model=384, block_size=256)
    defaults = {"token_budget": 50_000_000, "batch_size": 32}
    return TrainConfig(model=model, **{**defaults, **overrides})


STUDIES: dict[str, Study] = {
    "norm_placement": Study(
        name="norm_placement",
        hypothesis=(
            "Post-norm trains less stably at six layers and ends at higher validation "
            "loss than pre-norm, because post-norm renormalises the residual stream at "
            "every layer while pre-norm leaves it untouched from embedding to output."
        ),
        conditions={
            "pre": {"norm_placement": "pre"},
            "post": {"norm_placement": "post"},
        },
    ),
    "pos_encoding": Study(
        name="pos_encoding",
        hypothesis=(
            "The three are close at the training context of 256. Beyond it, learned "
            "encodings collapse because those positions were never trained, sinusoidal "
            "degrades but stays defined, and RoPE degrades most gracefully because its "
            "attention scores depend only on relative position."
        ),
        conditions={
            "learned": {"pos_encoding": "learned"},
            "sinusoidal": {"pos_encoding": "sinusoidal"},
            "rope": {"pos_encoding": "rope"},
        },
        # Not parameter-matched, and cannot be: sinusoidal and RoPE carry no
        # position parameters at all while learned carries n_positions * d_model.
        # Stated in FINDINGS rather than papered over.
        shared={"pos_capacity": A2_MAX_EVAL_CONTEXT},
    ),
    "head_count": Study(
        name="head_count",
        hypothesis=(
            "One head is clearly worse; returns diminish sharply past six. Parameter "
            "count is identical across all four conditions because heads only reshape a "
            "fixed d_model, so any difference is attributable to attention structure "
            "rather than capacity."
        ),
        conditions={
            "h1": {"n_head": 1},
            "h3": {"n_head": 3},
            "h6": {"n_head": 6},
            "h12": {"n_head": 12},
        },
    ),
}


# ---------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "run"])
    parser.add_argument(
        "--study", action="append", default=None, help="repeatable; omit to run all"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--token-budget", type=int, default=None, help="override for smoke runs")
    parser.add_argument("--seeds", type=int, default=None, help="override the seed count")
    parser.add_argument("--device", default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    selected = _select(args.study)

    if args.command == "list":
        for study in selected:
            print(
                f"{study.name}  ({len(study.conditions)} conditions x {len(study.seeds)} seeds "
                f"= {len(study)} runs)"
            )
            print(f"  conditions: {', '.join(study.conditions)}")
            print(f"  hypothesis: {study.hypothesis}\n")
        print(f"total: {sum(len(s) for s in selected)} runs")
        return 0

    if args.seeds is not None:
        selected = [replace(s, seeds=tuple(range(args.seeds))) for s in selected]

    base = sweep_base(
        load_meta(args.data_dir)["vocab_size"],
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        device=args.device,
        log_wandb=args.wandb,
        **({"token_budget": args.token_budget} if args.token_budget else {}),
    )

    records = run_sweep(selected, base, resume=not args.no_resume)
    return 0 if all(r["status"] == "completed" for r in records) else 1


def _select(names: list[str] | None) -> list[Study]:
    if not names:
        return list(STUDIES.values())
    unknown = [n for n in names if n not in STUDIES]
    if unknown:
        raise SystemExit(f"unknown study {unknown}; available: {sorted(STUDIES)}")
    return [STUDIES[n] for n in names]


if __name__ == "__main__":
    raise SystemExit(main())
