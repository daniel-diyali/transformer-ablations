"""Shared fixtures.

`trained_run` builds a complete miniature project — corpus, tokenizer, and a
finished training run — so the sampling and analysis layers can be tested
against real artifacts rather than hand-written stand-ins that might drift
from what training actually writes.
"""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from minigpt.data import DataConfig, build_tokenizer, encode_split
from minigpt.model import GPTConfig
from minigpt.train import TrainConfig, train

DOCS = [
    "Once upon a time a small cat sat on a warm mat in the sun.",
    "The dog ran to the park and played with a big red ball.",
    "Lily found a smooth stone by the river and showed her mother.",
    "Tom built a tall tower of blocks and then knocked it down.",
] * 60


@pytest.fixture(scope="session")
def mini_corpus(tmp_path_factory):
    """A tokenized corpus small enough to build in a second."""
    data_dir = tmp_path_factory.mktemp("data")
    shard = data_dir / "shard.parquet"
    pq.write_table(pa.table({"text": DOCS}), shard)

    cfg = DataConfig(vocab_size=512, data_dir=data_dir, tokenizer_sample_docs=len(DOCS))
    tokenizer = build_tokenizer(cfg, [shard])
    n_train = encode_split(tokenizer, [shard], cfg.bin_path("train"), cfg)
    n_val = encode_split(tokenizer, [shard], cfg.bin_path("val"), cfg)

    cfg.meta_path.write_text(
        json.dumps(
            {
                "vocab_size": tokenizer.get_vocab_size(),
                "n_train_tokens": n_train,
                "n_val_tokens": n_val,
                "eot_token_id": tokenizer.token_to_id("<|endoftext|>"),
            }
        )
    )
    return cfg


@pytest.fixture(scope="session")
def trained_run(mini_corpus, tmp_path_factory):
    """A finished run directory: config.json, metrics.jsonl, checkpoint, DONE."""
    out_dir = tmp_path_factory.mktemp("runs")
    cfg = TrainConfig(
        model=GPTConfig(
            vocab_size=mini_corpus.vocab_size,
            n_layer=2,
            n_head=2,
            d_model=32,
            block_size=16,
        ),
        data_dir=mini_corpus.data_dir,
        out_dir=out_dir,
        run_name="tiny",
        token_budget=6_144,
        batch_size=4,
        eval_interval_tokens=2_048,
        eval_batches=2,
        device="cpu",
        seed=0,
    )
    train(cfg)
    return cfg.run_dir()
