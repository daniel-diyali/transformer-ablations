"""Data pipeline specification.

Builds a small synthetic corpus as parquet and runs the real download-free
code path over it, so CI exercises tokenizer training, encoding, memmap
loading and batching without touching the network.
"""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from minigpt.data import (
    END_OF_TEXT,
    DataConfig,
    build_tokenizer,
    download_shards,
    encode_split,
    get_batch,
    iter_texts,
    load_tokens,
)

DOCS = [
    "Once upon a time there was a little cat who liked to sit in the sun.",
    "The dog ran fast to the park and played with a red ball all day long.",
    "Lily found a shiny stone near the river and showed it to her mother.",
    "Tom and Sam built a tall tower out of blocks and then knocked it down.",
] * 40


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A tokenized synthetic corpus: parquet in, uint16 tokens out."""
    root = tmp_path_factory.mktemp("corpus")
    shard = root / "shard.parquet"
    pq.write_table(pa.table({"text": DOCS}), shard)

    cfg = DataConfig(vocab_size=512, data_dir=root, tokenizer_sample_docs=len(DOCS))
    tokenizer = build_tokenizer(cfg, [shard])
    n_tokens = encode_split(tokenizer, [shard], cfg.bin_path("train"), cfg)

    cfg.meta_path.write_text(
        json.dumps({"vocab_size": tokenizer.get_vocab_size(), "n_train_tokens": n_tokens})
    )
    return cfg, tokenizer, n_tokens


def test_parquet_streaming_yields_every_document(corpus):
    cfg, _, _ = corpus
    assert len(list(iter_texts([cfg.data_dir / "shard.parquet"]))) == len(DOCS)


def test_iter_texts_respects_limit(corpus):
    cfg, _, _ = corpus
    assert len(list(iter_texts([cfg.data_dir / "shard.parquet"], limit=7))) == 7


def test_tokenizer_round_trip_is_lossless(corpus):
    """FR2.4. Byte-level BPE, so this must hold for every document."""
    _, tokenizer, _ = corpus
    for doc in DOCS[:4]:
        assert tokenizer.decode(tokenizer.encode(doc).ids) == doc


def test_tokenizer_round_trip_survives_unseen_characters(corpus):
    """Byte-level means coverage does not depend on what was in the corpus."""
    _, tokenizer, _ = corpus
    weird = "Ünïcodé — 漢字 — 🙂 — {}[]<>"
    assert tokenizer.decode(tokenizer.encode(weird).ids) == weird


def test_tokenizer_stays_within_the_vocab_cap(corpus):
    cfg, tokenizer, _ = corpus
    assert tokenizer.get_vocab_size() <= cfg.vocab_size


def test_token_file_size_matches_reported_count(corpus):
    """uint16 is two bytes per token; any mismatch means a truncated write."""
    cfg, _, n_tokens = corpus
    assert cfg.bin_path("train").stat().st_size == n_tokens * 2


def test_every_document_is_followed_by_an_end_of_text_token(corpus):
    cfg, tokenizer, _ = corpus
    tokens = np.fromfile(cfg.bin_path("train"), dtype=np.uint16)
    eot = tokenizer.token_to_id(END_OF_TEXT)
    assert int((tokens == eot).sum()) == len(DOCS)
    assert tokens[-1] == eot


def test_tokens_are_uint16_and_within_vocab(corpus):
    cfg, tokenizer, _ = corpus
    tokens = np.fromfile(cfg.bin_path("train"), dtype=np.uint16)
    assert tokens.dtype == np.uint16
    assert int(tokens.max()) < tokenizer.get_vocab_size()


def test_load_tokens_memmaps_the_expected_length(corpus):
    cfg, _, n_tokens = corpus
    assert load_tokens("train", cfg.data_dir).size == n_tokens


def test_load_tokens_rejects_a_corpus_that_disagrees_with_its_metadata(corpus, tmp_path):
    """A truncated rebuild must fail loudly, not silently change what a run trains on."""
    cfg, _, n_tokens = corpus
    (tmp_path / "train.bin").write_bytes(cfg.bin_path("train").read_bytes()[:-64])
    (tmp_path / "meta.json").write_text(json.dumps({"n_train_tokens": n_tokens}))

    with pytest.raises(RuntimeError, match="disagree"):
        load_tokens("train", tmp_path)


def test_load_tokens_names_the_fix_when_files_are_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"minigpt\.data prepare"):
        load_tokens("train", tmp_path)


def test_get_batch_returns_the_requested_shape_and_dtype(corpus):
    cfg, _, _ = corpus
    x, y = get_batch(load_tokens("train", cfg.data_dir), batch_size=4, block_size=8)
    assert x.shape == y.shape == (4, 8)
    # int64 because embedding lookup requires it; uint16 would raise.
    assert x.dtype == y.dtype == torch.int64


def test_targets_are_inputs_shifted_by_one(corpus):
    """The next-token objective. Off-by-one here trains the model on nothing."""
    cfg, _, _ = corpus
    x, y = get_batch(load_tokens("train", cfg.data_dir), batch_size=4, block_size=8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_batches_are_reproducible_per_seed(corpus):
    """Ablation conditions must see identical data in identical order."""
    cfg, _, _ = corpus
    tokens = load_tokens("train", cfg.data_dir)

    def sample(seed):
        return get_batch(tokens, 4, 8, generator=torch.Generator().manual_seed(seed))[0]

    assert torch.equal(sample(0), sample(0))
    assert not torch.equal(sample(0), sample(1))


def test_get_batch_refuses_a_corpus_shorter_than_one_block(corpus):
    with pytest.raises(ValueError, match="at least"):
        get_batch(np.arange(4, dtype=np.uint16), batch_size=1, block_size=8)


@pytest.mark.parametrize("n_shards", [0, 5])
def test_download_rejects_an_out_of_range_shard_count(n_shards):
    """Caught before any network call, so a typo fails instantly."""
    with pytest.raises(ValueError, match=r"must be 1\.\.4"):
        download_shards(DataConfig(n_train_shards=n_shards))
