"""Thermal pacing: does it cool the machine without touching the science?

The feature's whole claim is that pacing changes *when* arithmetic happens and
never *what* arithmetic happens. That claim is the only reason it is safe to
pace a sweep whose first runs were recorded at full speed, so it is asserted
here directly rather than trusted: same seed, paced and unpaced, identical
losses to the last bit.

The second group of tests guards the corollary — that a profile stays out of a
run's config fingerprint. If pacing leaked into the fingerprint, resuming a
paced sweep would refuse to skip the runs already finished at full speed, and
a night of compute would be repeated for no scientific reason.
"""

import pytest

from minigpt.experiments import RunSpec, Study, config_hash, run_sweep
from minigpt.model import GPTConfig
from minigpt.thermal import DEFAULT_PROFILE, PROFILES, ThermalProfile, get_profile
from minigpt.train import TrainConfig, _config_to_dict, train

PACED = ThermalProfile(name="test-paced", pause_every_steps=2, pause_seconds=0.05)


def tiny_config(mini_corpus, out_dir, **overrides) -> TrainConfig:
    defaults = dict(
        model=GPTConfig(
            vocab_size=mini_corpus.vocab_size, n_layer=1, n_head=2, d_model=24, block_size=16
        ),
        data_dir=mini_corpus.data_dir,
        out_dir=out_dir,
        token_budget=1_024,
        batch_size=4,
        eval_interval_tokens=512,
        eval_batches=1,
        device="cpu",
        seed=0,
    )
    return TrainConfig(**{**defaults, **overrides})


# ------------------------------------------------------- the invariance claim


def test_pacing_leaves_the_training_result_bit_identical(mini_corpus, tmp_path):
    """The central claim: idle gaps cost wall-clock and nothing else.

    Compared to the last bit rather than with a tolerance. Pacing does not
    reorder or re-associate a single floating-point operation, so "close
    enough" would be a weaker claim than the code actually supports — and
    would hide exactly the kind of drift this test exists to catch.
    """
    full = train(
        tiny_config(mini_corpus, tmp_path / "full", run_name="r"), thermal=PROFILES["full"]
    )
    paced = train(tiny_config(mini_corpus, tmp_path / "paced", run_name="r"), thermal=PACED)

    assert paced["best_val_loss"] == full["best_val_loss"]
    assert paced["final_val_loss"] == full["final_val_loss"]
    assert (paced["step"], paced["tokens_seen"]) == (full["step"], full["tokens_seen"])


def test_pacing_costs_wall_clock_and_reports_it_separately(mini_corpus, tmp_path):
    """Idle time is recorded, so pacing cannot be mistaken for a slowdown."""
    paced = train(tiny_config(mini_corpus, tmp_path / "paced", run_name="r"), thermal=PACED)

    assert paced["paused_s"] > 0
    assert paced["thermal_profile"] == "test-paced"
    # Throughput excluding deliberate idle must beat wall-clock throughput,
    # which is the whole point of reporting both.
    assert paced["compute_tokens_per_s"] > paced["tokens_per_s"]


def test_an_unpaced_run_reports_no_idle_time(mini_corpus, tmp_path):
    full = train(
        tiny_config(mini_corpus, tmp_path / "full", run_name="r"), thermal=PROFILES["full"]
    )

    assert full["paused_s"] == 0
    assert full["thermal_profile"] == "full"


def test_a_run_defaults_to_no_pacing(mini_corpus, tmp_path):
    """Library callers and tests get full speed unless they ask otherwise."""
    record = train(tiny_config(mini_corpus, tmp_path / "default", run_name="r"))

    assert record["thermal_profile"] == "full"
    assert record["paused_s"] == 0


# ------------------------------------------- pacing is not part of the science


def test_a_paced_rerun_still_skips_runs_finished_at_full_speed(mini_corpus, tmp_path, capsys):
    """The corollary that protects a night of compute.

    This sweep's first three runs were recorded at full speed and the rest are
    being paced. If a profile reached the config fingerprint, resume would call
    that a config change and rerun everything already done.
    """
    study = Study(
        name="smoke",
        hypothesis="pacing is invisible to resume",
        conditions={"pre": {"norm_placement": "pre"}, "post": {"norm_placement": "post"}},
        seeds=(0,),
    )
    out_dir = tmp_path / "runs"
    base = tiny_config(mini_corpus, out_dir)

    run_sweep([study], base, thermal=PROFILES["full"])
    capsys.readouterr()
    run_sweep([study], base, thermal=get_profile("quiet"))

    # Skipped, not rerun, and not raised as a fingerprint mismatch.
    assert capsys.readouterr().out.count("already done, skipping") == 2


def test_a_profile_is_not_an_input_to_the_fingerprint(mini_corpus, tmp_path):
    """Stated at the source: the fingerprint is computed from the config alone."""
    config = tiny_config(mini_corpus, tmp_path / "runs")
    spec = RunSpec(study="smoke", condition="pre", seed=0, config=config)

    assert spec.fingerprint == config_hash(config)
    assert "thermal" not in _config_to_dict(config)


# ----------------------------------------------------------- profile plumbing


def test_the_default_profile_paces():
    """A long unattended sweep must be sustainable unless full speed is asked for."""
    assert get_profile(DEFAULT_PROFILE).paces


def test_full_speed_does_not_pace():
    assert not PROFILES["full"].paces


def test_an_unknown_profile_is_rejected_by_name():
    with pytest.raises(ValueError, match="unknown thermal profile"):
        get_profile("chilly")


@pytest.mark.parametrize("field", ["pause_every_steps", "pause_seconds", "cooldown_between_runs_s"])
def test_a_negative_setting_is_rejected(field):
    """A negative pause would sleep forever or not at all; neither silently."""
    with pytest.raises(ValueError, match=field):
        ThermalProfile(name="bad", **{field: -1})


def test_a_half_specified_profile_does_not_pace():
    """Either lever at zero disables pacing, rather than pausing for 0s forever."""
    assert not ThermalProfile(name="no-seconds", pause_every_steps=4).paces
    assert not ThermalProfile(name="no-steps", pause_seconds=0.5).paces


def test_pausing_happens_only_on_due_steps():
    profile = ThermalProfile(name="every-third", pause_every_steps=3, pause_seconds=0.01)

    assert profile.pause_if_due(step=1, device="cpu") == 0.0
    assert profile.pause_if_due(step=2, device="cpu") == 0.0
    assert profile.pause_if_due(step=3, device="cpu") == pytest.approx(0.01)


def test_duty_cycle_predicts_the_slowdown_before_committing_hours():
    """0.6s of idle after every 4 steps of 0.26s is a ~63% duty cycle."""
    cool = get_profile("cool")

    assert cool.expected_duty_cycle(step_seconds=0.26) == pytest.approx(0.634, abs=0.01)
    assert PROFILES["full"].expected_duty_cycle(step_seconds=0.26) == 1.0
    # A nonsense step time cannot produce a nonsense prediction.
    assert cool.expected_duty_cycle(step_seconds=0.0) == 1.0
