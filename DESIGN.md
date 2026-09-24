# DESIGN — transformer-ablations

**Status:** signed off 2026-09-24
**Drafted:** 2026-09-24
**Satisfies:** REQUIREMENTS.md

## 1. Shape of the system

Five layers, each with one job. Data flows in one direction.

```
corpus ──▶ tokenizer ──▶ train.bin / val.bin (uint16 memmap)
                                  │
                                  ▼
  config ──▶ trainer ──▶ run directory (metrics.jsonl, ckpt.pt, config.json)
                                  │
              sweep runner ───────┤  (N conditions × M seeds)
                                  ▼
                          results aggregator ──▶ figures/ + FINDINGS.md
```

The important boundary is between **running experiments** and **analyzing them**. The
trainer writes plain files and knows nothing about ablations. The analysis layer reads
those files and knows nothing about PyTorch. That split means a failed sweep doesn't cost
the analysis work, and charts can be regenerated without retraining.

## 2. Components

All code lives in the `minigpt/` package. Corpus and run artifacts live in `data/` and
`runs/` at the repo root, both gitignored — the package and the artifacts are kept apart
so a module name never collides with a directory full of `.bin` files.

### 2.1 `minigpt/data.py` — corpus to tokens

Responsibility: turn raw text into a flat token array, once.

- `build_tokenizer` — trains BPE on a corpus sample, writes `tokenizer.json`.
- `prepare` — streams the corpus, encodes, appends to a `uint16` binary.
  Writes `train.bin`, `val.bin`, and `meta.json` (vocab size, token counts, tokenizer
  hash, split seed).

`uint16` caps vocabulary at 65,536, which is far above the planned 8,192. Storing tokens
flat rather than as padded sequences means a batch is just `N` random offsets into the
array — no padding, no attention-mask bookkeeping, and every token in a batch contributes
to the loss.

The split is by document, not by token offset, so validation text never appears as a
training-sequence suffix.

### 2.2 `minigpt/model.py` — the transformer

Responsibility: the architecture, and nothing else. No I/O, no logging, no globals.

```python
@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int
    n_layer: int
    n_head: int
    d_model: int
    block_size: int          # training context length
    dropout: float = 0.0
    bias: bool = True
    tie_weights: bool = True
    norm_placement: Literal["pre", "post"] = "pre"
    pos_encoding: Literal["learned", "sinusoidal", "rope"] = "learned"
```

```python
class GPT(nn.Module):
    def forward(self, idx, targets=None) -> tuple[Tensor, Tensor | None]
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None) -> Tensor
    def num_params(self, non_embedding: bool = True) -> int
```

Every ablation dimension is a field on `GPTConfig`. This is the load-bearing decision of
the whole design: it means an ablation is a config change, conditions cannot silently
diverge in unrelated ways, and one set of tests covers all conditions.

Attention is computed explicitly — QKV projection, scaled dot product, causal mask,
softmax, weighted sum, output projection — because the point of the project is showing
the mechanism. A flag may route to `F.scaled_dot_product_attention` **only** as a speed
comparison and a numerical-equivalence test target, never as the default path.

### 2.3 `minigpt/train.py` — the training loop

Responsibility: turn a config into a trained checkpoint and a metrics file.

Hand-written loop: batch sampling, forward, loss, backward, gradient clipping, optimizer
step, LR schedule, gradient accumulation, periodic eval, checkpointing.

Budgeting is in **tokens processed**, not steps or epochs. Two conditions with different
batch shapes must still see identical data volume, or the comparison is meaningless.

AdamW with two parameter groups: weight decay applies to matrices, not to biases or
layer-norm parameters. Applying decay to norm gains is a common quiet bug that shifts
results.

### 2.4 `minigpt/experiments.py` — the sweep runner

Responsibility: expand a study definition into runs, execute them, skip completed ones.

A study is declarative:

```python
Study(
    name="norm_placement",
    hypothesis="Post-norm trains less stably at this depth and ends at higher "
               "validation loss; the gap widens without LR warmup.",
    base=SWEEP_CONFIG,
    conditions={"pre": {"norm_placement": "pre"},
                "post": {"norm_placement": "post"}},
    seeds=(0, 1, 2),
)
```

Run identity is `hash(config + seed)`. A run whose directory contains a `DONE` marker is
skipped, which makes an interrupted overnight sweep resumable without bookkeeping.

### 2.5 `minigpt/analysis.py` — results to figures

Responsibility: read run directories, emit charts and tables. Pure post-processing.

Plots show **individual seeds plus the mean**, never a bare mean line. If three seeds
disagree about which condition wins, the chart must make that visible. That honesty is
the differentiator; hiding variance behind a mean is the thing most public ablation repos
get wrong.

## 3. Data contracts

**`meta.json`** — `vocab_size`, `n_train_tokens`, `n_val_tokens`, `tokenizer_sha`,
`split_seed`.

**`metrics.jsonl`** — one object per eval interval:
```json
{"step": 500, "tokens": 4096000, "train_loss": 3.41, "val_loss": 3.52,
 "lr": 0.00058, "elapsed_s": 141.2, "tokens_per_s": 29040}
```

**`config.json`** — the full resolved config plus provenance: git SHA, device, torch
version, seed, hostname.

**`results.jsonl`** — one row per completed run, written by the sweep runner: study,
condition, seed, final and best validation loss, tokens, wall-clock, run directory.

Plain JSONL over a database because runs are append-only, the volume is small, the files
are diffable and readable, and a half-finished sweep leaves valid data.

## 4. Failure handling

Fail loudly at the boundary. No silent recovery.

| Failure | Response |
|---|---|
| Loss becomes NaN or Inf | Abort the run immediately, write `FAILED` with the step and last good loss. A diverged run is data, not an error to hide — post-norm may legitimately diverge, and that *is* the finding. |
| Loss climbs for N consecutive evals | Log a divergence warning, continue. Recorded in the run record. |
| Resume with a config that doesn't match the checkpoint | Refuse to start. Mismatched resume silently corrupts a comparison. |
| `train.bin` missing or `meta.json` inconsistent | Fail at startup with the command needed to rebuild. |
| MPS unavailable | Fall back to CPU with a loud warning; record the device in provenance so slow runs are explicable later. |
| Sweep run crashes | Mark that run failed, continue the sweep, report failures in the summary. One bad condition must not cost the night. |
| OOM | Fail with the config's memory estimate and the suggested `batch_size` / `grad_accum` split. |

## 5. Trade-offs

### 5.1 Compute target

| Option | Pros | Cons |
|---|---|---|
| **Local MPS (M4 Pro, 24 GB)** — *recommended* | Unlimited uninterrupted runtime; sweeps run overnight; no upload/re-upload; consistent hardware makes timings comparable; $0 | Slower per step than a T4/A100; MPS has occasional operator gaps |
| Colab/Kaggle free GPU | Faster raw throughput; CUDA is better trodden | Session limits and disconnects are hostile to 20+ sequential runs; hardware varies between sessions, polluting timing comparisons |
| Hybrid | Best of both | Two environments to keep correct |

**Recommendation: local MPS, with device-agnostic code.** The bottleneck here is *number
of comparable runs*, not speed of one run. 24 GB of unified memory comfortably holds a
14M-parameter model. Writing for `cuda|mps|cpu` from the start costs nothing and leaves
the door open to burst a longer flagship run onto a borrowed GPU later.

**Confirmed by measurement (2026-09-24, day-1 spike).** torch 2.14.0, M4 Pro, fp32,
explicit attention, 40 timed steps after warmup:

| Config | Params | Throughput | ms/step | Peak mem |
|---|---|---|---|---|
| Planned d384 / L6 / H6, batch 32 | 13.9M | 23,500 tok/s | 349 | 3.5 GB |
| Same, batch 64 | 13.9M | 23,900 tok/s | 684 | 5.9 GB |
| A3 worst case, H12 | 13.9M | 21,200 tok/s | 387 | 3.8 GB |
| A2 long-context eval, 512 ctx | 14.0M | 20,700 tok/s | 396 | 3.8 GB |
| Fallback d256 / H8 | 6.9M | 36,500 tok/s | 224 | 3.6 GB |

Batch 64 buys ~2% throughput for 68% more memory, so **batch 32 is the setting** —
the device is already saturated at 32. The 12-head condition costs only 10%, so A3 is
cheap. Memory peaks at 3.5 GB against 24 GB available, leaving ample headroom.

Every op the design depends on passed on MPS: `scaled_dot_product_attention` (needed as
the equivalence-test target), complex64 multiply (the RoPE complex formulation),
bf16 and fp16 autocast, `cross_entropy`, `clip_grad_norm_`, and `multinomial`. The
fallback config exists only as insurance and is not needed.

The spike measured *speed only*. It trains on one fixed random batch and says nothing
about correctness — that is M2's job.

### 5.2 Tokenizer

| Option | Pros | Cons |
|---|---|---|
| Character-level | Zero dependencies; trivially correct | Wastes context; samples look worse; not what real models do |
| **BPE via `tokenizers`** — *recommended* | Realistic; fast; small vocab fits TinyStories well | One dependency |
| BPE implemented by hand | More "from scratch" credit | Days of work on the least interesting part of the project |

**Recommendation: HuggingFace `tokenizers`, vocab 8,192.** "From scratch" is a claim
about the model and training loop. Spending a week on merge tables buys nothing an
interviewer cares about.

### 5.3 Config and sweeps

| Option | Pros | Cons |
|---|---|---|
| **Frozen dataclass + CLI overrides + Python study defs** — *recommended* | No framework to learn; configs are type-checked; studies are readable code | Hand-rolled sweep expansion (~60 lines) |
| Hydra | Powerful composition | Heavy for one repo; obscures what's happening |
| Bare YAML | Simple | No types, no validation; typos become silent wrong runs |

**Recommendation: dataclass.** A frozen dataclass hashes cleanly for run identity and
catches typos at construction rather than 40 minutes into a run.

### 5.4 Logging

**Recommendation: JSONL as the source of truth, Weights & Biases as an optional backend
behind a flag.** The repo must work fully with no account and no network. W&B is one
`if cfg.wandb:` block, and it puts named industry observability tooling on the resume —
a gap flagged in the job-posting analysis. Pending Q3.

## 6. Ablation selection

Three studies recommended, chosen for what they let you *say*, not just measure.

**A1 — Layer-norm placement (pre vs post).**
Hypothesis: post-norm trains less stably at this depth and lands at higher validation
loss; removing warmup widens the gap. This is why every modern decoder is pre-norm, and
explaining that from your own data is a strong interview answer. Cheap: 2 conditions.

**A2 — Positional encoding (learned vs sinusoidal vs RoPE).** *The headline study.*
Hypothesis: the three are close at the training context, but at evaluation contexts
longer than training, learned encodings collapse while RoPE degrades gracefully.
The chart — validation loss vs evaluation context length, with a vertical line at the
training context — is a genuinely interesting figure, not a rehash. It is also the most
current: RoPE is what essentially every model shipping today uses. 3 conditions, plus a
cheap extra eval pass at 2× and 4× context.

**A3 — Head count at fixed width (1 / 3 / 6 / 12 heads, `d_model` = 384).**
Hypothesis: 1 head is clearly worse; returns diminish sharply past 6; parameter count is
identical across conditions, so any difference is attributable to attention structure
alone. The controlled-parameter framing is what makes this rigorous rather than
confounded. 4 conditions.

**A4 (stretch) — LR schedule: cosine + warmup vs constant.** Real but well-trodden. Run
only if the sweep budget allows, and it pairs naturally with A1's no-warmup arm.

Total: 9 conditions × 3 seeds = 27 sweep runs, plus 1 flagship.

An honest caveat that belongs in the writeup: conclusions drawn at 14M parameters and a
reduced token budget may not hold at scale. Saying so demonstrates more understanding
than claiming universality.

## 7. Model and budget

**Sweep config** (for all ablation runs — small and fast so comparisons are affordable):

| | |
|---|---|
| Layers / heads / `d_model` | 6 / 6 / 384 |
| Context | 256 |
| Vocab | 8,192 |
| Batch size | 32 (measured: the device saturates here) |
| Parameters | ≈ 13.9M (≈ 10.7M non-embedding) |
| Token budget | **50M — confirmed, not provisional** |

At the measured 23,500 tok/s, a 50M-token run takes **35 minutes**, inside the 45-minute
NFR3 ceiling. The full 27-run sweep is **~16 hours**, which is two overnight sessions.

**Flagship config** — same architecture at the best-performing settings, trained on a
200M-token budget: **2.4 hours**. Roughly Chinchilla-adjacent for this size; the sweep
runs are deliberately under-trained, which is fine for comparison but is a caveat worth
stating.

Total project compute is therefore about **18.5 hours**, all of it overnight.

Batch shape is chosen to hold tokens-per-step constant across conditions, using gradient
accumulation to absorb any memory differences.

Not yet measured: whether bf16 autocast beats fp32 on MPS. It runs, but the throughput
gain is unknown and MPS gains are often modest. Treated as an optional M3 optimization,
never as a correctness-affecting change mid-sweep.

## 8. Dependencies

`torch`, `numpy`, `tokenizers`, `datasets` (corpus fetch only), `matplotlib`, `pytest`,
`ruff`. Optional: `wandb`.

Python 3.11+ via `uv` — the system Python here is 3.9.6, which is too old for current
torch. `uv` is already installed.

Deliberately absent: HuggingFace `transformers`, Lightning, Hydra, Accelerate. Their
presence would undercut the claim the project exists to make.

## 9. Test strategy

Tests are the specification. The ones that matter:

- **Causality.** Changing token `t` must not alter logits at any position `< t`. Run by
  perturbation, not by inspecting the mask — this catches a wrong mask *and* a wrong
  application of a right mask.
- **Attention equivalence.** The hand-rolled attention must match
  `F.scaled_dot_product_attention(is_causal=True)` within float tolerance. This is the
  single highest-value test in the repo: it proves the from-scratch implementation is
  correct rather than merely plausible.
- **Shapes and parameter count** across every ablation config combination, so no
  condition is quietly broken.
- **Overfit one batch.** Loss on a single batch must drop below 0.1 within N steps. If
  this fails, the training loop is wrong, and it catches almost every plumbing bug.
- **Loss at init** ≈ `ln(vocab_size)` ≈ 9.01. A wrong initialization or a mis-scaled
  logit path shows up here immediately.
- **Checkpoint round-trip.** Save, reload, and confirm identical logits and optimizer
  state.
- **Tokenizer round-trip.** `decode(encode(x)) == x`.
- **Determinism.** Same seed, same config → same loss sequence on the same device.

All tests run on CPU with tiny configs, in under a minute, on every PR.
