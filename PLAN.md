# PLAN — transformer-ablations

**Status:** signed off 2026-09-24
**Drafted:** 2026-09-24
**Target:** complete by 2026-10-15, ahead of the October–November application wave.

Eight milestones. Each is independently verifiable, lands as its own branch and PR, and
leaves `main` working. Nothing merges without tests passing.

## Risk register — read before starting

Three unknowns that can move the plan. All are probed in week 1 rather than discovered in
week 3.

| Risk | Impact | Status | Resolution |
|---|---|---|---|
| ~~**MPS throughput unknown**~~ | Sets every token budget and the sweep schedule | **Closed 2026-09-24** | Measured 23,500 tok/s at the planned config. The 50M-token sweep budget holds — 35 min/run, inside the 45-min ceiling. Full sweep ~16 h, flagship 2.4 h, ~18.5 h total. No shrink needed; the `d256` fallback is unnecessary. |
| ~~**MPS operator gaps or numerical quirks**~~ | Could force CPU (far too slow) or CUDA | **Closed 2026-09-24** | Every op the design needs passed: SDPA, complex64 multiply for RoPE, bf16/fp16 autocast, `cross_entropy`, `clip_grad_norm_`, `multinomial`. No CUDA fallback required. |
| **Post-norm may diverge, not merely underperform** | A1's chart becomes a divergence story | Open until M6 | Already designed for: a NaN run is recorded as `FAILED` with its last good loss. Divergence *is* the finding — the writeup frames it that way. |

**Day-1 spike: done.** Lived in `spikes/` (gitignored), probed ten ops and benchmarked
five configs. Both compute risks closed before any architecture was committed, which was
the point. Measured numbers are in DESIGN §5.1 and §7. Two incidental findings: batch 64
buys ~2% throughput for 68% more memory, so batch 32 is the setting; and the 12-head A3
condition costs only 10%, so that study is cheap.

The spike measured speed only — it trains on one fixed random batch and validates nothing
about correctness. M2 is where correctness gets established.

---

## M0 — Scaffold and CI
**Branch:** `chore/scaffold` · **~half a day**

Repo skeleton, `pyproject.toml` with pinned dependencies, `uv` environment on Python
3.11, `ruff` config, `pytest` config, `.gitignore` covering `data/`, `runs/`, `spikes/`,
`*.bin`, `*.pt`. GitHub Actions workflow running lint and tests on every PR.

**Verified by:** a placeholder test passes locally and in CI; `ruff check` is clean; a
clean clone reaches a working environment with one documented command.

**Gate:** CI green on the PR.

---

## M1 — Data pipeline
**Branch:** `feat/data-pipeline` · **~1.5 days**

Fetch TinyStories, train the BPE tokenizer (vocab 8,192), stream-encode to
`train.bin` / `val.bin` as `uint16`, write `meta.json`. Document-level split with a fixed
seed. A `get_batch` helper that memory-maps and samples random offsets.

**Verified by:** tokenizer round-trip test on sampled documents; `meta.json` token counts
match file sizes exactly; `get_batch` returns correct shapes and dtype; a decoded random
batch is readable English; split determinism holds across two runs.

**Gate:** corpus tokenized, counts reported in the PR description.

---

## M2 — Model
**Branch:** `feat/gpt-model` · **~3 days** · *the correctness milestone*

`GPTConfig` and `GPT`: embeddings, all three positional-encoding modes, causal
multi-head attention written from primitives, MLP blocks, pre/post norm placement,
weight tying, initialization, `generate`.

Written to be read — this file is the one an interviewer will open.

**Verified by:** the full test list in DESIGN §9. The two that gate the merge:
- attention matches `F.scaled_dot_product_attention(is_causal=True)` within tolerance
- causality by perturbation — editing token `t` leaves all logits before `t` untouched

Plus loss-at-init ≈ `ln(8192)` ≈ 9.01, and shape/param-count coverage across every
ablation config combination.

**Gate:** all correctness tests green. Untrained model, but *provably correct*.

---

## M3 — Training loop and throughput
**Branch:** `feat/training-loop` · **~2.5 days**

Hand-written loop: token-budgeted training, AdamW with correct no-decay groups, gradient
clipping, cosine schedule with warmup, gradient accumulation, periodic eval,
checkpoint save/resume with RNG state, JSONL metrics, provenance capture, NaN abort.

**Verified by:** overfit-one-batch drives loss below 0.1; checkpoint round-trip reproduces
logits and optimizer state; resume reproduces an uninterrupted loss curve; same seed
reproduces the same loss sequence; a deliberately broken LR triggers the NaN abort path.

**Gate:** throughput of the *real* training loop reported in the PR and compared against
the spike's 23,500 tok/s. The spike measured a stripped-down loop; if the real one is
materially slower, the gap is explained before proceeding rather than absorbed silently
into the schedule. Budgets in DESIGN §7 are already set from measurement, so this is a
confirmation, not a decision.

---

## M4 — Baseline run and sampling
**Branch:** `feat/baseline-run` · **~1 day plus training time**

Train the sweep config once, end to end. Implement temperature and top-k sampling.
Produce the first loss curve and the first text samples.

**Verified by:** validation loss decreases smoothly and plateaus; samples are recognizable
English sentences; run directory contains complete metrics, config, and checkpoint.

**Gate:** a loss curve and a sample paragraph in the PR description. First shareable
artifact — from here the project is already resume-legible even if later milestones slip.

---

## M5 — Experiment harness
**Branch:** `feat/sweep-runner` · **~2 days**

`Study` definitions, config expansion over conditions × seeds, content-addressed run IDs,
`DONE` markers for skip-on-rerun, per-run failure isolation, `results.jsonl` aggregation,
a sweep summary report. Optional W&B backend behind a flag, pending Q3.

**Verified by:** a 2-condition × 2-seed smoke study on a 30-second budget completes;
re-running skips finished runs; an injected mid-sweep crash leaves prior results intact
and the sweep continues; `results.jsonl` parses and matches the run directories.

**Gate:** smoke sweep reproducible from a clean clone.

---

## M6 — Ablations
**Branch:** `feat/ablations` · **~2 days of work, ~2–3 nights of compute**

Run A1 (norm placement), A2 (positional encoding, including long-context evaluation), and
A3 (head count). 27 runs. A4 only if the budget allows.

Analysis layer: per-study charts plotting individual seeds plus the mean, a summary table,
and the A2 headline figure — validation loss vs evaluation context length, with the
training context marked.

**Verified by:** every condition has ≥3 completed seeds; charts regenerate from
`results.jsonl` without retraining; seed spread is visible in every figure; any failed or
diverged run is reported rather than dropped.

**Gate:** all three studies complete with charts.

---

## M7 — Flagship run, findings, README
**Branch:** `docs/findings` · **~1.5 days plus training time**

Flagship training run at the best configuration on the larger token budget. Write
`FINDINGS.md`: per study — hypothesis, method, result, interpretation, and explicitly what
contradicted expectations. README with setup, reproduction commands for every figure, and
honest hardware and wall-clock costs.

**Verified by:** a stranger can clone, install, and reproduce any figure from the README
alone; every claim in `FINDINGS.md` traces to a chart or a number in `results.jsonl`.

**Gate:** Daniel's review. This is the artifact recruiters actually read.

---

## Sequence

| Week | Milestones | Ends with |
|---|---|---|
| 1 (Sep 24 – Oct 1) | Spike, M0, M1, M2 | A provably correct transformer, untrained |
| 2 (Oct 2 – Oct 8) | M3, M4, M5 | A trained model, samples, working sweep harness |
| 3 (Oct 9 – Oct 15) | M6, M7 | Three ablation studies, charts, findings, public repo |

Training runs happen overnight, so compute time overlaps the next day's work rather than
blocking it.

**If time runs short,** cut in this order: A4, then A3, then the flagship run — keeping
three seeds per condition throughout. Two rigorous studies beat four sloppy ones, and
seed count is the thing that makes the work credible. M4 is the point after which the
project is worth putting on a resume regardless of what follows.

## Working agreements

- Branch per milestone, PR per branch, `gh pr create --fill`, tests and `ruff` run before
  every commit.
- Nothing merges to `main` without Daniel's approval.
- Decisions and their reasons land in `NOTES.md` as they happen.
- `data/`, `runs/`, and checkpoints stay out of git. Only `results.jsonl` and figures are
  committed, so the repo stays small and the findings stay diffable.
- DESIGN.md gets updated when reality diverges from it — particularly the M3 budget
  revision.

## Sign-off

Signed off by Daniel on 2026-09-24. All five open questions resolved — see
REQUIREMENTS §7. Ablations are A1 + A2 + A3, corpus is TinyStories, W&B goes in behind an
optional flag, and the repo is public from the start.

Nothing is blocking. Next action is the day-1 throughput spike, then M0.
