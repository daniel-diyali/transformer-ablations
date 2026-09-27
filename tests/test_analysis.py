"""Analysis specification.

The analysis layer reads what a run wrote and emits figures. It imports no
torch, which is what lets every chart be regenerated without retraining —
and lets these tests run in milliseconds.
"""

import json

import pytest

from minigpt.analysis import (
    load_context_eval,
    load_metrics,
    load_record,
    load_sweep,
    main,
    plot_context_scaling,
    plot_loss_curve,
    plot_study,
    study_summary,
    summarize,
)


def test_metrics_are_read_in_logged_order(trained_run):
    entries = load_metrics(trained_run)
    assert entries
    assert [e["tokens"] for e in entries] == sorted(e["tokens"] for e in entries)


def test_every_entry_carries_the_fields_charts_depend_on(trained_run):
    required = {"step", "tokens", "train_loss", "val_loss", "lr", "elapsed_s", "tokens_per_s"}
    assert all(required <= set(entry) for entry in load_metrics(trained_run))


def test_missing_metrics_name_the_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="no logged metrics"):
        load_metrics(tmp_path)


def test_an_empty_metrics_file_is_rejected(tmp_path):
    (tmp_path / "metrics.jsonl").write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_metrics(tmp_path)


def test_a_finished_run_has_a_done_record(trained_run):
    record = load_record(trained_run)
    assert record is not None and record["status"] == "completed"


def test_an_unfinished_run_has_no_record(tmp_path):
    assert load_record(tmp_path) is None


def test_the_loss_curve_is_written_to_disk(trained_run, tmp_path):
    out = plot_loss_curve(trained_run, out_path=tmp_path / "loss.png")
    assert out.exists() and out.stat().st_size > 1_000


def test_plotting_creates_the_output_directory(trained_run, tmp_path):
    out = plot_loss_curve(trained_run, out_path=tmp_path / "nested" / "deep" / "loss.png")
    assert out.exists()


def test_the_chart_can_be_regenerated_without_retraining(trained_run, tmp_path):
    """The point of separating analysis from training."""
    first = plot_loss_curve(trained_run, out_path=tmp_path / "a.png")
    second = plot_loss_curve(trained_run, out_path=tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes()


def test_the_summary_reports_the_numbers_worth_quoting(trained_run):
    summary = summarize(trained_run)
    entries = load_metrics(trained_run)

    assert summary["status"] == "completed"
    assert summary["tokens"] == entries[-1]["tokens"]
    assert summary["final_val_loss"] == entries[-1]["val_loss"]
    assert summary["best_val_loss"] == min(e["val_loss"] for e in entries)
    assert summary["best_val_loss"] <= summary["final_val_loss"]


def test_the_summary_reports_that_training_reduced_loss(trained_run):
    summary = summarize(trained_run)
    assert summary["final_val_loss"] < summary["first_val_loss"]


def test_the_cli_writes_a_chart(trained_run, tmp_path, capsys):
    out = tmp_path / "cli.png"
    assert main(["loss-curve", "--run", str(trained_run), "--out", str(out)]) == 0
    assert out.exists()
    assert str(out) in capsys.readouterr().out


def test_the_cli_prints_a_json_summary(trained_run, capsys):
    assert main(["summary", "--run", str(trained_run)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


# -------------------------------------------------------- sweep-level charts


@pytest.fixture
def sweep_dir(tmp_path):
    """A synthetic sweep: two conditions, three seeds, plus context evaluations."""
    results = []
    for condition, losses in [("pre", [1.880, 1.884, 1.890]), ("post", [1.950, 1.958, 1.962])]:
        for seed, loss in enumerate(losses):
            results.append(
                {
                    "study": "norm_placement",
                    "condition": condition,
                    "seed": seed,
                    "run": f"norm_placement-{condition}-s{seed}",
                    "status": "completed",
                    "final_val_loss": loss,
                }
            )
    # A failed run must not be counted in any summary.
    results.append(
        {
            "study": "norm_placement",
            "condition": "post",
            "seed": 9,
            "run": "norm_placement-post-s9",
            "status": "error",
            "final_val_loss": None,
            "reason": "injected",
        }
    )
    (tmp_path / "results.jsonl").write_text("\n".join(json.dumps(r) for r in results) + "\n")

    context_rows = []
    for condition in ("learned", "rope"):
        for seed in range(2):
            for context in (256, 512):
                unrepresentable = condition == "learned" and context == 512
                context_rows.append(
                    {
                        "study": "pos_encoding",
                        "condition": condition,
                        "seed": seed,
                        "context": context,
                        "train_context": 256,
                        "val_loss": None if unrepresentable else 1.9 + 0.1 * (context // 256),
                        "status": "unrepresentable" if unrepresentable else "ok",
                    }
                )
    (tmp_path / "context_eval.jsonl").write_text(
        "\n".join(json.dumps(r) for r in context_rows) + "\n"
    )
    return tmp_path


def test_a_sweep_without_results_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="no sweep has been run"):
        load_sweep(tmp_path)


def test_the_summary_reports_mean_and_spread_per_condition(sweep_dir):
    summary = {
        row["condition"]: row for row in study_summary(load_sweep(sweep_dir), "norm_placement")
    }

    assert summary["pre"]["n_seeds"] == 3
    assert summary["pre"]["mean"] == pytest.approx((1.880 + 1.884 + 1.890) / 3)
    assert summary["pre"]["spread"] == pytest.approx(0.010)
    assert summary["pre"]["losses"] == sorted(summary["pre"]["losses"])


def test_failed_runs_are_excluded_from_the_summary(sweep_dir):
    """A failed run has no loss; averaging it in would corrupt the comparison."""
    summary = {
        row["condition"]: row for row in study_summary(load_sweep(sweep_dir), "norm_placement")
    }
    assert summary["post"]["n_seeds"] == 3


def test_summarizing_an_unknown_study_raises(sweep_dir):
    with pytest.raises(ValueError, match="no completed runs"):
        study_summary(load_sweep(sweep_dir), "nonexistent")


def test_the_study_chart_is_written(sweep_dir, tmp_path):
    out = plot_study(load_sweep(sweep_dir), "norm_placement", out_path=tmp_path / "s.png")
    assert out.exists() and out.stat().st_size > 1_000


def test_the_study_chart_regenerates_identically(sweep_dir, tmp_path):
    records = load_sweep(sweep_dir)
    first = plot_study(records, "norm_placement", out_path=tmp_path / "a.png")
    second = plot_study(records, "norm_placement", out_path=tmp_path / "b.png")
    assert first.read_bytes() == second.read_bytes()


def test_missing_context_evaluations_name_the_command_that_makes_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="context-eval"):
        load_context_eval(tmp_path)


def test_the_context_chart_handles_conditions_that_cannot_reach_a_length(sweep_dir, tmp_path):
    """Learned encodings have no embeddings past their capacity. That is the finding."""
    rows = load_context_eval(sweep_dir)
    assert any(r["val_loss"] is None for r in rows)

    out = plot_context_scaling(rows, out_path=tmp_path / "ctx.png")
    assert out.exists() and out.stat().st_size > 1_000


def test_the_cli_writes_a_study_chart(sweep_dir, tmp_path, capsys):
    out = tmp_path / "cli-study.png"
    assert (
        main(["study", "--sweep", str(sweep_dir), "--study", "norm_placement", "--out", str(out)])
        == 0
    )
    assert out.exists()
    capsys.readouterr()


def test_the_cli_requires_a_study_name_for_the_study_chart(sweep_dir):
    with pytest.raises(SystemExit, match="--study is required"):
        main(["study", "--sweep", str(sweep_dir)])


def test_the_cli_prints_a_table_with_seed_values(sweep_dir, capsys):
    assert main(["table", "--sweep", str(sweep_dir)]) == 0
    out = capsys.readouterr().out
    assert "norm_placement" in out and "spread" in out and "1.8800" in out


def test_the_cli_requires_a_run_for_single_run_commands(sweep_dir):
    with pytest.raises(SystemExit, match="--run is required"):
        main(["summary"])
