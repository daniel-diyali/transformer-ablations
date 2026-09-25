"""Load a trained run and generate text from it.

Reconstructs the model from the config the run recorded, rather than from
arguments supplied here, so a checkpoint can never be loaded into an
architecture it was not trained with.

    python -m minigpt.sample --run runs/baseline-50M --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from minigpt.data import DataConfig, load_tokenizer
from minigpt.model import GPT, GPTConfig
from minigpt.train import CHECKPOINT, pick_device


def load_run(run_dir: Path, device: str | None = None) -> tuple[GPT, dict]:
    """Rebuild the model a run trained, with its weights loaded.

    The architecture comes from the run's own config.json. Passing model
    dimensions here instead would let a caller quietly load weights into the
    wrong shape, or the right shape with the wrong positional encoding — which
    loads without complaint and generates nonsense.
    """
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / CHECKPOINT
    for path in (config_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; {run_dir} is not a finished run")

    saved = json.loads(config_path.read_text())
    model_config = GPTConfig(**saved["config"]["model"])

    device = pick_device(device)
    model = GPT(model_config).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model, saved


def sample(
    run_dir: Path,
    prompt: str = "",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 50,
    seed: int = 0,
    device: str | None = None,
    data_dir: Path | None = None,
) -> str:
    """Generate one continuation of `prompt`."""
    model, saved = load_run(run_dir, device)
    tokenizer = load_tokenizer(DataConfig(data_dir=data_dir or Path(saved["config"]["data_dir"])))

    torch.manual_seed(seed)
    device = next(model.parameters()).device

    # An empty prompt still needs a first token: use end-of-text, which is
    # what the model saw before the start of every training document.
    ids = tokenizer.encode(prompt).ids if prompt else [tokenizer.token_to_id("<|endoftext|>")]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="a run directory under runs/")
    parser.add_argument("--prompt", default="", help="empty prompt starts a fresh document")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    for i in range(args.num_samples):
        text = sample(
            args.run,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + i,
            device=args.device,
            data_dir=args.data_dir,
        )
        if args.num_samples > 1:
            print(f"--- sample {i + 1} (seed {args.seed + i}) ---")
        print(text)
        if i < args.num_samples - 1:
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
