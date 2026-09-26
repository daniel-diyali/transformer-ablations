"""Thermal pacing: run the same experiment, cooler.

A 27-run sweep pins an Apple GPU at 98-99% for hours. On a laptop that means
sustained fan noise and heat while charging, which is a real cost to the
person whose machine it is.

The lever chosen here is **duty cycling**: do the identical work, but insert
short idle gaps so the GPU is not saturated continuously. That choice is
deliberate, because of what it does *not* touch.

Pacing changes when arithmetic happens, never what arithmetic happens. Batch
size, learning rate, data order and gradients are all untouched, so a paced
run and a full-speed run from the same seed produce bit-identical losses —
asserted directly in the tests. The alternative levers (smaller batch,
gradient accumulation) would reduce memory but *would* change results, and
would invalidate the runs already recorded at batch 32.

For that reason a `ThermalProfile` is **not** part of `TrainConfig` and takes
no part in a run's config fingerprint. Pacing is an execution concern, not a
scientific one; a run paced differently is the same experiment, and the sweep
must still recognise it as already done.

    python -m minigpt.experiments run --thermal cool
"""

from __future__ import annotations

import shutil
import subprocess
import time
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ThermalProfile:
    """How hard to let the machine work.

    `pause_seconds` of idle after every `pause_every_steps` optimizer steps.
    Zero for either disables pacing entirely.
    """

    name: str
    pause_every_steps: int = 0
    pause_seconds: float = 0.0
    cooldown_between_runs_s: float = 0.0
    # Hold work while unplugged. A multi-hour sweep will flatten a laptop
    # battery and take the machine down with it, losing the run in progress.
    pause_on_battery: bool = False
    battery_poll_seconds: float = 30.0
    # Releases cached MPS blocks. Lowers the allocated footprint without
    # touching any result, since it frees cache rather than live tensors.
    empty_cache_every_steps: int = 0

    def __post_init__(self) -> None:
        for field_name in ("pause_every_steps", "empty_cache_every_steps"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0, got {getattr(self, field_name)}")
        for field_name in ("pause_seconds", "cooldown_between_runs_s"):
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be >= 0, got {getattr(self, field_name)}")

    @property
    def paces(self) -> bool:
        return self.pause_every_steps > 0 and self.pause_seconds > 0

    def expected_duty_cycle(self, step_seconds: float) -> float:
        """Fraction of wall time spent computing, given a measured step time.

        Useful for predicting the slowdown before committing hours to it.
        """
        if not self.paces or step_seconds <= 0:
            return 1.0
        work = self.pause_every_steps * step_seconds
        return work / (work + self.pause_seconds)

    def pause_if_due(self, step: int, device: str) -> float:
        """Idle briefly if this step is a pause point. Returns seconds slept.

        Synchronises first. MPS queues work asynchronously, so sleeping
        without a barrier would just let the GPU chew through the backlog —
        the process would look idle while the hardware stayed hot.
        """
        if not self.paces or step % self.pause_every_steps != 0:
            return 0.0
        _synchronize(device)
        time.sleep(self.pause_seconds)
        return self.pause_seconds

    def release_cache_if_due(self, step: int, device: str) -> None:
        if self.empty_cache_every_steps <= 0 or step % self.empty_cache_every_steps != 0:
            return
        if device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    def wait_for_mains(self, device: str) -> float:
        """Block while the machine is on battery. Returns seconds waited.

        Idle time, like every other lever here: it delays arithmetic without
        changing it, so a run interrupted by an unplugged laptop still
        produces the same result it would have produced plugged in.
        """
        if not self.pause_on_battery or not on_battery():
            return 0.0

        _synchronize(device)
        started = time.monotonic()
        print("on battery — holding until power is reconnected", flush=True)
        while on_battery():
            time.sleep(self.battery_poll_seconds)
        waited = time.monotonic() - started
        print(f"power back after {waited / 60:.1f} min — resuming", flush=True)
        return waited

    def cool_between_runs(self, device: str) -> float:
        """Idle between runs, letting the machine shed heat before the next one."""
        if self.cooldown_between_runs_s <= 0:
            return 0.0
        _synchronize(device)
        time.sleep(self.cooldown_between_runs_s)
        return self.cooldown_between_runs_s

    def as_dict(self) -> dict:
        return asdict(self)


def on_battery() -> bool:
    """True when running unplugged.

    Best-effort and deliberately fail-open: if the power state cannot be read
    — a non-macOS host, a missing tool, a timeout — this reports mains power
    so a sweep is never blocked by a broken probe.
    """
    if shutil.which("pmset") is None:
        return False
    try:
        result = subprocess.run(
            ["pmset", "-g", "ps"], capture_output=True, text=True, timeout=5, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return "Battery Power" in result.stdout


def _synchronize(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


# Measured on an Apple M4 Pro at ~0.26 s/step (batch 32, block 256). Duty
# cycles quoted are predictions from that step time; DESIGN records what was
# actually observed.
PROFILES: dict[str, ThermalProfile] = {
    # Everything the hardware will give. Fastest, hottest, loudest.
    "full": ThermalProfile(name="full"),
    # The default for long sweeps: noticeably cooler, still finishes overnight.
    "cool": ThermalProfile(
        name="cool",
        # Coarse on purpose. Sub-second gaps hold the same duty cycle but never
        # let the GPU drop its clocks, so average power barely moves. A pause
        # measured in seconds gives it long enough to actually idle down.
        pause_every_steps=40,
        pause_seconds=6.0,
        cooldown_between_runs_s=60.0,
        empty_cache_every_steps=40,
        pause_on_battery=True,
    ),
    # For working at the machine while a sweep runs behind you.
    "quiet": ThermalProfile(
        name="quiet",
        pause_every_steps=40,
        pause_seconds=20.0,
        cooldown_between_runs_s=120.0,
        empty_cache_every_steps=40,
        pause_on_battery=True,
    ),
}

DEFAULT_PROFILE = "cool"


def get_profile(name: str) -> ThermalProfile:
    if name not in PROFILES:
        raise ValueError(f"unknown thermal profile {name!r}; available: {sorted(PROFILES)}")
    return PROFILES[name]
