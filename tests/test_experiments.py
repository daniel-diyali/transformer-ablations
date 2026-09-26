"""Sweep specification.

Two groups matter here. The first covers the machinery: expansion, resume,
and failure isolation. The second checks the studies against the claims they
make about themselves — if A3 is not parameter-matched, its conclusion is
confounded, and a test is the only thing that will ever notice.
"""

import json
from dataclasses import replace

import pytest

from minigpt.experiments import (
    STUDIES,
    RunSpec,
    Study,
    apply_overrides,
    config_hash,
    load_results,
    main,
    run_sweep,
    sweep_base,
)
from minigpt.model import GPT, GPTConfig
from minigpt.train import TrainConfig

TINY_MODEL = dict(vocab_size=512, n_layer=2, n_head=2, d_model=32, block_size=16)


def tiny_base(mini_corpus, out_dir, **overrides) -> TrainConfig:
    defaults = dict(
        model=GPTConfig(**{**TINY_MODEL, "vocab_size": mini_corpus.vocab_size}),
        data_dir=mini_corpus.data_dir,
        out_dir=out_dir,
        token_budget=1_024,
        batch_size=4,
        eval_interval_tokens=1_024,
        eval_batches=1,
        device="cpu",
    )
    return TrainConfig(**{**defaults, **overrides})


TWO_BY_TWO = Study(
    name="smoke",
    hypothesis="pre-norm ends lower than post-norm",
    conditions={"pre": {"norm_placement": "pre"}, "post": {"norm_placement": "post"}},
    seeds=(0, 1),
)


# ------------------------------------------------------------- overrides


def test_overrides_route_to_the_config_that_owns_the_field():
    base = TrainConfig(model=GPTConfig(**TINY_MODEL))
    updated = apply_overrides(base, {"n_head": 4, "lr": 1e-3})
    assert updated.model.n_head == 4
    assert updated.lr == 1e-3


def test_an_unknown_override_raises_rather_than_being_ignored():
    """The most dangerous possible failure in this file.

    A dropped override would let the sweep finish with every condition
    identical, and the chart would read as a genuine finding of "no
    difference" rather than a bug.
    """
    base = TrainConfig(model=GPTConfig(**TINY_MODEL))
    with pytest.raises(ValueError, match="unknown override"):
        apply_overrides(base, {"norm_placemnt": "post"})


def test_overrides_leave_the_base_config_untouched():
    base = TrainConfig(model=GPTConfig(**TINY_MODEL))
    apply_overrides(base, {"n_head": 4})
    assert base.model.n_head == TINY_MODEL["n_head"]


def test_the_fingerprint_distinguishes_configs_and_is_stable():
    base = TrainConfig(model=GPTConfig(**TINY_MODEL))
    pre = apply_overrides(base, {"norm_placement": "pre"})
    post = apply_overrides(base, {"norm_placement": "post"})

    assert config_hash(pre) != config_hash(post)
    assert config_hash(pre) == config_hash(apply_overrides(base, {"norm_placement": "pre"}))


# ----------------------------------------------------------------- studies


def test_expansion_covers_every_condition_and_seed():
    base = TrainConfig(model=GPTConfig(**TINY_MODEL))
    specs = list(TWO_BY_TWO.expand(base))

    assert len(specs) == len(TWO_BY_TWO) == 4
    assert {s.name for s in specs} == {
        "smoke-pre-s0",
        "smoke-pre-s1",
        "smoke-post-s0",
        "smoke-post-s1",
    }


def test_expansion_applies_the_condition_and_the_seed():
    base = TrainConfig(model=GPTConfig(**TINY_MODEL))
    specs = {s.name: s for s in TWO_BY_TWO.expand(base)}

    assert specs["smoke-post-s1"].config.model.norm_placement == "post"
    assert specs["smoke-post-s1"].config.seed == 1
    assert specs["smoke-pre-s0"].config.model.norm_placement == "pre"


def test_shared_settings_reach_every_condition():
    base = TrainConfig(model=GPTConfig(**TINY_MODEL))
    study = replace(TWO_BY_TWO, shared={"dropout": 0.1})
    assert all(s.config.model.dropout == 0.1 for s in study.expand(base))


def test_a_study_needs_something_to_compare():
    with pytest.raises(ValueError, match="at least two conditions"):
        Study(name="x", hypothesis="h", conditions={"only": {}})


def test_a_study_needs_at_least_one_seed():
    with pytest.raises(ValueError, match="at least one seed"):
        Study(name="x", hypothesis="h", conditions={"a": {}, "b": {}}, seeds=())


def test_duplicate_seeds_are_rejected():
    """Two identical runs would look like agreement between independent seeds."""
    with pytest.raises(ValueError, match="duplicate seeds"):
        Study(name="x", hypothesis="h", conditions={"a": {}, "b": {}}, seeds=(0, 0, 1))


# --------------------------------------------- the studies' own claims


def test_every_defined_study_expands_into_buildable_models():
    """Seeds do not change architecture, so one build per condition suffices."""
    base = sweep_base(8192)
    for study in STUDIES.values():
        for spec in study.expand(base):
            if spec.seed == study.seeds[0]:
                GPT(spec.config.model)  # raises if any condition is invalid


def test_the_sweep_is_twenty_seven_runs():
    assert sum(len(s) for s in STUDIES.values()) == 27


@pytest.mark.parametrize("study_name", ["norm_placement", "head_count"])
def test_a1_and_a3_are_parameter_matched(study_name):
    """Their conclusions depend on it.

    A1 rearranges where normalisation sits; A3 reshapes a fixed d_model into
    more or fewer heads. Neither changes capacity, so any difference in loss
    is attributable to the thing under study. If this fails, that study is
    confounded and its result means nothing.
    """
    base = sweep_base(8192)
    totals = {
        GPT(spec.config.model).num_params(non_embedding=False)
        for spec in STUDIES[study_name].expand(base)
        if spec.seed == 0
    }
    assert len(totals) == 1, f"{study_name} conditions differ in capacity: {sorted(totals)}"


def test_a2_is_not_parameter_matched_and_that_is_expected():
    """Sinusoidal and RoPE carry no position parameters; learned carries many.

    The asymmetry is inherent to the comparison rather than an oversight.
    This test exists so the fact stays visible and lands in the writeup.
    """
    base = sweep_base(8192)
    totals = {
        spec.condition: GPT(spec.config.model).num_params(non_embedding=False)
        for spec in STUDIES["pos_encoding"].expand(base)
        if spec.seed == 0
    }
    assert totals["sinusoidal"] == totals["rope"]
    assert totals["learned"] > totals["sinusoidal"]


def test_a2_can_be_evaluated_past_its_training_context():
    """Without the extra capacity the long-context arm has no data point."""
    base = sweep_base(8192)
    for spec in STUDIES["pos_encoding"].expand(base):
        assert spec.config.model.n_positions > spec.config.model.block_size


def test_every_study_states_a_hypothesis():
    assert all(len(s.hypothesis) > 40 for s in STUDIES.values())


# ------------------------------------------------------------- the sweep


def test_a_smoke_sweep_completes_and_records_every_run(mini_corpus, tmp_path):
    base = tiny_base(mini_corpus, tmp_path / "runs")
    records = run_sweep([TWO_BY_TWO], base)

    assert len(records) == 4
    assert all(r["status"] == "completed" for r in records)
    assert all(r["final_val_loss"] is not None for r in records)


def test_results_are_written_as_they_land(mini_corpus, tmp_path):
    """An interrupted sweep must still leave its finished runs on disk."""
    base = tiny_base(mini_corpus, tmp_path / "runs")
    run_sweep([TWO_BY_TWO], base)

    lines = (tmp_path / "runs" / "results.jsonl").read_text().splitlines()
    assert len(lines) == 4
    assert all(json.loads(line)["run"] for line in lines)


def test_each_record_points_at_a_real_run_directory(mini_corpus, tmp_path):
    base = tiny_base(mini_corpus, tmp_path / "runs")
    for record in run_sweep([TWO_BY_TWO], base):
        assert (tmp_path / "runs" / record["run"] / "DONE.json").exists()


def test_rerunning_skips_finished_runs(mini_corpus, tmp_path):
    base = tiny_base(mini_corpus, tmp_path / "runs")
    run_sweep([TWO_BY_TWO], base)
    before = (tmp_path / "runs" / "results.jsonl").read_text()

    run_sweep([TWO_BY_TWO], base)
    assert (tmp_path / "runs" / "results.jsonl").read_text() == before


def test_a_changed_config_under_the_same_name_is_refused(mini_corpus, tmp_path):
    """Otherwise one study name would hold results from two configurations."""
    base = tiny_base(mini_corpus, tmp_path / "runs")
    run_sweep([TWO_BY_TWO], base)

    with pytest.raises(RuntimeError, match="different config"):
        run_sweep([TWO_BY_TWO], tiny_base(mini_corpus, tmp_path / "runs", lr=9e-4))


def test_one_failing_run_does_not_stop_the_sweep(mini_corpus, tmp_path, monkeypatch):
    """A bad condition must cost one run, not a night of compute."""
    import minigpt.experiments as experiments

    real_train = experiments.train
    calls = {"n": 0}

    def flaky(cfg, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected failure")
        return real_train(cfg, **kwargs)

    monkeypatch.setattr(experiments, "train", flaky)
    records = run_sweep([TWO_BY_TWO], tiny_base(mini_corpus, tmp_path / "runs"))

    statuses = [r["status"] for r in records]
    assert statuses.count("error") == 1
    assert statuses.count("completed") == 3
    assert "injected failure" in next(r["reason"] for r in records if r["status"] == "error")


def test_a_failed_run_is_recorded_rather_than_dropped(mini_corpus, tmp_path, monkeypatch):
    import minigpt.experiments as experiments

    monkeypatch.setattr(
        experiments, "train", lambda cfg, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    run_sweep([TWO_BY_TWO], tiny_base(mini_corpus, tmp_path / "runs"))

    saved = load_results(tmp_path / "runs")
    assert len(saved) == 4
    assert all(r["status"] == "error" for r in saved.values())


def test_load_results_is_empty_before_any_sweep(tmp_path):
    assert load_results(tmp_path) == {}


def test_a_run_spec_names_itself_from_study_condition_and_seed():
    base = TrainConfig(model=GPTConfig(**TINY_MODEL))
    spec = RunSpec("study", "cond", 7, base)
    assert spec.name == "study-cond-s7"


# ----------------------------------------------------------------- the cli


def test_the_cli_lists_studies_with_their_hypotheses(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "norm_placement" in out and "hypothesis" in out
    assert "total: 27 runs" in out


def test_the_cli_can_select_one_study(capsys):
    main(["list", "--study", "head_count"])
    out = capsys.readouterr().out
    assert "head_count" in out and "norm_placement" not in out


def test_the_cli_rejects_an_unknown_study():
    with pytest.raises(SystemExit, match="unknown study"):
        main(["list", "--study", "nope"])


def test_the_cli_runs_a_shrunken_sweep(mini_corpus, tmp_path, capsys):
    """Shrinking the model and the budget is what makes wiring testable in seconds."""
    exit_code = main(
        [
            "run",
            "--study",
            "norm_placement",
            "--token-budget",
            "512",
            "--seeds",
            "1",
            "--n-layer",
            "1",
            "--n-head",
            "2",
            "--d-model",
            "24",
            "--block-size",
            "16",
            "--data-dir",
            str(mini_corpus.data_dir),
            "--out-dir",
            str(tmp_path / "runs"),
            "--device",
            "cpu",
        ]
    )
    assert exit_code == 0
    assert "sweep finished" in capsys.readouterr().out
    assert len(load_results(tmp_path / "runs")) == 2
