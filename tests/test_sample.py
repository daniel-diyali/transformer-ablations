"""Sampling specification.

The property that matters here is that a checkpoint is reloaded into the
architecture it was trained with, not one supplied by the caller. Loading
weights into a plausible-but-wrong architecture succeeds silently and
generates nonsense, which is far more expensive than a crash.
"""

import json

import pytest
import torch

from minigpt.sample import load_run, main, sample


def test_loading_a_run_restores_the_architecture_it_was_trained_with(trained_run):
    model, saved = load_run(trained_run, device="cpu")
    recorded = saved["config"]["model"]

    assert model.cfg.n_layer == recorded["n_layer"]
    assert model.cfg.d_model == recorded["d_model"]
    assert model.cfg.pos_encoding == recorded["pos_encoding"]
    assert model.cfg.norm_placement == recorded["norm_placement"]


def test_a_loaded_model_carries_the_trained_weights(trained_run):
    """Not a freshly initialised model that merely has the right shape."""
    model, _ = load_run(trained_run, device="cpu")
    checkpoint = torch.load(trained_run / "ckpt.pt", map_location="cpu", weights_only=False)
    saved_weight = checkpoint["model"]["token_embedding.weight"]
    assert torch.equal(model.token_embedding.weight, saved_weight)


def test_a_loaded_model_is_in_eval_mode(trained_run):
    """Sampling with dropout active would add noise to every generated token."""
    model, _ = load_run(trained_run, device="cpu")
    assert not model.training


def test_loading_a_directory_without_a_checkpoint_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError, match="not a finished run"):
        load_run(tmp_path)


def test_loading_a_run_missing_its_config_fails_clearly(trained_run, tmp_path):
    (tmp_path / "ckpt.pt").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        load_run(tmp_path)


def test_sampling_produces_text_that_extends_the_prompt(trained_run):
    text = sample(trained_run, prompt="The dog", max_new_tokens=20, device="cpu")
    assert text.startswith("The dog")
    assert len(text) > len("The dog")


def test_sampling_works_without_a_prompt(trained_run):
    """An empty prompt starts from end-of-text, as every training document did."""
    text = sample(trained_run, prompt="", max_new_tokens=20, device="cpu")
    assert isinstance(text, str) and text


def test_sampling_is_reproducible_for_a_given_seed(trained_run):
    kwargs = dict(prompt="The dog", max_new_tokens=20, device="cpu")
    assert sample(trained_run, seed=0, **kwargs) == sample(trained_run, seed=0, **kwargs)


def test_different_seeds_give_different_samples(trained_run):
    kwargs = dict(prompt="The dog", max_new_tokens=40, temperature=1.0, device="cpu")
    assert sample(trained_run, seed=0, **kwargs) != sample(trained_run, seed=1, **kwargs)


def test_greedy_sampling_ignores_the_seed(trained_run):
    """temperature=0 takes the argmax, so randomness cannot enter."""
    kwargs = dict(prompt="The dog", max_new_tokens=20, temperature=0.0, device="cpu")
    assert sample(trained_run, seed=0, **kwargs) == sample(trained_run, seed=99, **kwargs)


def test_the_cli_prints_a_sample(trained_run, capsys):
    exit_code = main(["--run", str(trained_run), "--max-new-tokens", "10", "--device", "cpu"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip()


def test_the_cli_can_print_several_samples(trained_run, capsys):
    main(
        [
            "--run",
            str(trained_run),
            "--max-new-tokens",
            "10",
            "--num-samples",
            "3",
            "--device",
            "cpu",
        ]
    )
    assert capsys.readouterr().out.count("--- sample") == 3


def test_the_run_config_records_the_data_directory_sampling_needs(trained_run):
    """Sampling finds the tokenizer through the run's own config."""
    saved = json.loads((trained_run / "config.json").read_text())
    assert "data_dir" in saved["config"]
