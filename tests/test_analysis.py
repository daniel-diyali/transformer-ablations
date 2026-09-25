"""Analysis specification.

The analysis layer reads what a run wrote and emits figures. It imports no
torch, which is what lets every chart be regenerated without retraining —
and lets these tests run in milliseconds.
"""

import json

import pytest

from minigpt.analysis import load_metrics, load_record, main, plot_loss_curve, summarize


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
