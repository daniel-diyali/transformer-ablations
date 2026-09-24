"""Corpus to tokens, once.

Downloads TinyStories, trains a BPE tokenizer, and encodes the corpus into a
flat `uint16` array that training memory-maps. Nothing here runs during
training — see DESIGN.md section 2.1.

Storing tokens flat rather than as padded sequences means a batch is just N
random offsets into the array: no padding, no attention-mask bookkeeping, and
every token in a batch contributes to the loss.

    python -m minigpt.data prepare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Marks a document boundary so the model can learn where stories end.
END_OF_TEXT = "<|endoftext|>"

REBUILD = "Run: python -m minigpt.data prepare"

# TinyStories ships four train shards and one validation shard. Listed
# explicitly rather than globbed so a corpus change is a visible diff.
TRAIN_SHARDS = (
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet",
    "data/train-00001-of-00004-5852b56a2bd28fd9.parquet",
    "data/train-00002-of-00004-a26307300439e943.parquet",
    "data/train-00003-of-00004-d243063613e5a057.parquet",
)
VAL_SHARD = "data/validation-00000-of-00001-869c898b519ad725.parquet"


@dataclass(frozen=True)
class DataConfig:
    repo_id: str = "roneneldan/TinyStories"
    vocab_size: int = 8192
    data_dir: Path = Path("data")
    # 1-4. All four give ~460M tokens, comfortably above the 200M flagship
    # budget. Fewer shards makes a faster dev loop.
    n_train_shards: int = 4
    # Docs sampled to fit the BPE merges. 200k is far more than an 8k vocab
    # over simple English needs, and it keeps tokenizer training under a minute.
    tokenizer_sample_docs: int = 200_000
    encode_batch_size: int = 10_000

    @property
    def tokenizer_path(self) -> Path:
        return self.data_dir / "tokenizer.json"

    @property
    def meta_path(self) -> Path:
        return self.data_dir / "meta.json"

    def bin_path(self, split: str) -> Path:
        return self.data_dir / f"{split}.bin"


# --------------------------------------------------------------- downloading


def download_shards(cfg: DataConfig) -> tuple[list[Path], Path]:
    """Fetch parquet shards from the Hub, using its local cache."""
    if not 1 <= cfg.n_train_shards <= len(TRAIN_SHARDS):
        raise ValueError(f"n_train_shards must be 1..{len(TRAIN_SHARDS)}, got {cfg.n_train_shards}")

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(repo_id=cfg.repo_id, repo_type="dataset", filename=name))

    train = [fetch(s) for s in TRAIN_SHARDS[: cfg.n_train_shards]]
    return train, fetch(VAL_SHARD)


def iter_texts(paths: list[Path], limit: int | None = None) -> Iterator[str]:
    """Stream documents out of parquet without loading a shard at once."""
    seen = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=10_000, columns=["text"]):
            for text in batch.column("text").to_pylist():
                if text:
                    yield text
                    seen += 1
                    if limit is not None and seen >= limit:
                        return


# --------------------------------------------------------------- tokenizer


def build_tokenizer(cfg: DataConfig, train_paths: list[Path]) -> Tokenizer:
    """Train byte-level BPE on a sample of the corpus.

    Byte-level means every input is representable, so decode(encode(x)) == x
    holds for arbitrary text rather than only for text the vocabulary covers.
    """
    # Byte-level BPE starts from all 256 byte values and adds the special
    # tokens, so no vocabulary smaller than that can exist.
    floor = 256 + 1  # 256 bytes + END_OF_TEXT
    if cfg.vocab_size < floor:
        raise ValueError(
            f"vocab_size={cfg.vocab_size} is below the byte-level floor of {floor} "
            f"(256 byte values plus {END_OF_TEXT})"
        )

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=[END_OF_TEXT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tokenizer.train_from_iterator(
        iter_texts(train_paths, limit=cfg.tokenizer_sample_docs),
        trainer=trainer,
        length=cfg.tokenizer_sample_docs,
    )

    actual = tokenizer.get_vocab_size()
    if actual > cfg.vocab_size:
        raise RuntimeError(f"tokenizer produced {actual} tokens, above the {cfg.vocab_size} cap")

    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(cfg.tokenizer_path))
    return tokenizer


def load_tokenizer(cfg: DataConfig) -> Tokenizer:
    if not cfg.tokenizer_path.exists():
        raise FileNotFoundError(f"{cfg.tokenizer_path} missing. {REBUILD}")
    return Tokenizer.from_file(str(cfg.tokenizer_path))


# ----------------------------------------------------------------- encoding


def encode_split(tokenizer: Tokenizer, paths: list[Path], out_path: Path, cfg: DataConfig) -> int:
    """Encode documents to a flat uint16 file, one EOT token between each.

    Written incrementally so corpus size is bounded by disk, not by RAM.
    """
    eot = tokenizer.token_to_id(END_OF_TEXT)
    if eot is None:
        raise RuntimeError(f"tokenizer has no {END_OF_TEXT} token")

    total = 0
    batch: list[str] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("wb") as fh:

        def flush(texts: list[str]) -> int:
            if not texts:
                return 0
            ids: list[int] = []
            for encoding in tokenizer.encode_batch(texts):
                ids.extend(encoding.ids)
                ids.append(eot)
            arr = np.asarray(ids, dtype=np.uint16)
            arr.tofile(fh)
            return arr.size

        for text in iter_texts(paths):
            batch.append(text)
            if len(batch) >= cfg.encode_batch_size:
                total += flush(batch)
                batch = []
                _progress(f"  {out_path.name}: {total:,} tokens")
        total += flush(batch)

    print(f"  {out_path.name}: {total:,} tokens", file=sys.stderr)
    return total


def prepare(cfg: DataConfig) -> dict:
    """Download, train the tokenizer, encode both splits, write meta.json."""
    print(f"downloading {cfg.n_train_shards} train shard(s) + validation", file=sys.stderr)
    train_paths, val_path = download_shards(cfg)

    print(f"training BPE (vocab {cfg.vocab_size})", file=sys.stderr)
    tokenizer = build_tokenizer(cfg, train_paths)

    print("encoding", file=sys.stderr)
    n_train = encode_split(tokenizer, train_paths, cfg.bin_path("train"), cfg)
    n_val = encode_split(tokenizer, [val_path], cfg.bin_path("val"), cfg)

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "n_train_tokens": n_train,
        "n_val_tokens": n_val,
        "eot_token_id": tokenizer.token_to_id(END_OF_TEXT),
        "tokenizer_sha": _sha256(cfg.tokenizer_path),
        "config": {**asdict(cfg), "data_dir": str(cfg.data_dir)},
    }
    cfg.meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def _progress(line: str) -> None:
    """Overwrite in place on a terminal; stay quiet when piped to a log."""
    if sys.stderr.isatty():
        print(line, end="\r", file=sys.stderr)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ------------------------------------------------------------------ loading


def load_meta(data_dir: Path = Path("data")) -> dict:
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing. {REBUILD}")
    return json.loads(meta_path.read_text())


def load_tokens(split: str, data_dir: Path = Path("data")) -> np.memmap:
    """Memory-map a split. Fails loudly if it disagrees with meta.json."""
    path = data_dir / f"{split}.bin"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing. {REBUILD}")

    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    expected = load_meta(data_dir).get(f"n_{split}_tokens")
    if expected is not None and tokens.size != expected:
        raise RuntimeError(
            f"{path} holds {tokens.size:,} tokens but meta.json claims {expected:,}. "
            f"The corpus and its metadata disagree. {REBUILD}"
        )
    return tokens


def get_batch(
    tokens: np.ndarray,
    batch_size: int,
    block_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a batch of (context, next-token) pairs at random offsets.

    `y` is `x` shifted one position, so every position in the block supplies a
    training signal.
    """
    if tokens.size < block_size + 1:
        raise ValueError(f"need at least {block_size + 1} tokens, corpus has {tokens.size}")

    high = tokens.size - block_size - 1
    ix = torch.randint(high, (batch_size,), generator=generator)

    # np.memmap slices are views; astype forces the copy torch.from_numpy needs.
    x = torch.stack([torch.from_numpy(tokens[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy(tokens[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------- cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "inspect"])
    parser.add_argument("--vocab-size", type=int, default=DataConfig.vocab_size)
    parser.add_argument("--shards", type=int, default=DataConfig.n_train_shards)
    parser.add_argument("--data-dir", type=Path, default=DataConfig.data_dir)
    args = parser.parse_args(argv)

    cfg = DataConfig(vocab_size=args.vocab_size, n_train_shards=args.shards, data_dir=args.data_dir)

    if args.command == "prepare":
        meta = prepare(cfg)
        print(json.dumps(meta, indent=2))
    else:
        meta = load_meta(cfg.data_dir)
        tokenizer = load_tokenizer(cfg)
        tokens = load_tokens("train", cfg.data_dir)
        print(json.dumps(meta, indent=2))
        print("\n--- decoded sample ---")
        print(tokenizer.decode(tokens[:200].astype(np.int64).tolist()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
