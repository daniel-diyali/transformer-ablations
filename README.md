# transformer-ablations

A GPT-style decoder-only transformer implemented from scratch in PyTorch, trained at
roughly 14M parameters, used to run **controlled ablations on architectural choices** —
with multiple seeds per condition and variance reported honestly.

> **Status: scaffolding done, model not yet written.**
> Design signed off, environment and CI green, throughput measured on the target
> hardware. No trained results yet. This section gets replaced with findings and charts
> as they land — see [PLAN.md](PLAN.md) for exactly where the project is.

## What this is

Implementing a transformer is commodity work; there are thousands of public repos that do
it. The point of this one is the **experimental method layered on top**:

- Every ablation dimension is a **config flag on a single model class**, never a code
  fork — so conditions cannot silently diverge in some unrelated way.
- Runs are budgeted in **tokens processed**, not epochs or steps, so conditions with
  different batch shapes still see identical data volume.
- **Every condition runs on at least three seeds.** Charts plot individual seeds
  alongside the mean. If the seeds disagree about which condition wins, the chart says so.
- Attention is computed from primitives and **tested for numerical equivalence** against
  `F.scaled_dot_product_attention`, so "from scratch" is a verified claim rather than an
  assertion.

## Planned studies

| Study | Question | Hypothesis |
|---|---|---|
| **A1 — norm placement** | Pre-norm vs post-norm | Post-norm trains less stably at this depth; the gap widens without LR warmup |
| **A2 — positional encoding** | Learned vs sinusoidal vs RoPE | Close at the training context; beyond it, learned encodings collapse while RoPE degrades gracefully |
| **A3 — head count** | 1 / 3 / 6 / 12 heads at fixed `d_model` | 1 head is clearly worse, returns diminish past 6 — parameter count held identical, so differences are attributable to attention structure alone |

## Planned setup

- **Model** — 6 layers, 6 heads, `d_model` 384, context 256, 8,192-token BPE vocabulary
  (≈ 13.9M parameters)
- **Corpus** — TinyStories, chosen so a model this small still produces coherent English
- **Hardware** — Apple M4 Pro (24 GB unified memory), MPS. Code is device-agnostic across
  `cuda | mps | cpu`
- **Cost** — $0. Local hardware only

Deliberately not used: HuggingFace `transformers`, Lightning, Hydra, Accelerate. Their
presence would undercut the claim the project exists to make.

## Documents

- [REQUIREMENTS.md](REQUIREMENTS.md) — what it must do, non-goals, assumptions
- [DESIGN.md](DESIGN.md) — components, data contracts, trade-offs, test strategy
- [PLAN.md](PLAN.md) — milestones, risk register, sequence
- [NOTES.md](NOTES.md) — decisions and why they were made

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11 && uv pip install -e ".[dev]"
uv run pytest        # 6 environment smoke tests, ~1s
uv run ruff check .
```

The smoke tests assert what this README claims about the environment — the Python and
torch floors, and the presence of the ops the model depends on — so a mismatch fails in
a second rather than forty minutes into a training run.

## Measured cost

Day-one benchmark on the target hardware, torch 2.14.0, fp32, explicit attention:

| | |
|---|---|
| Throughput | **23,500 tok/s** at d384 / L6 / H6, batch 32 |
| Peak memory | 3.5 GB of 24 GB available |
| One ablation run (50M tokens) | 35 minutes |
| Full 27-run sweep | ~16 hours |
| Flagship run (200M tokens) | 2.4 hours |
| **Total project compute** | **~18.5 hours**, all overnight |

Every op the design needs runs on MPS, including complex64 multiply for RoPE and both
autocast dtypes. No CUDA fallback is required.

## Reproduction

Commands for regenerating each figure land with the runs that produce them. Every figure
in the final writeup will be reproducible from a clean clone, with the hardware and
wall-clock cost stated honestly rather than omitted.
