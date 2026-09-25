"""Ablation studies: declare conditions, expand over seeds, run, aggregate.

A study says what it is testing and what it expects, then names the single
field that differs between its conditions. Everything else is inherited from
one shared base config, so two conditions cannot drift apart in some way
nobody intended.

    python -m minigpt.experiments run --study norm_placement
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields

from minigpt.model import GPTConfig
from minigpt.train import TrainConfig, _config_to_dict

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
