# PLAN — transformer-ablations

**Status:** draft, awaiting Daniel's sign-off
**Drafted:** 2026-09-24
**Target:** complete by 2026-10-15, ahead of the October–November application wave.

Eight milestones. Each is independently verifiable, lands as its own branch and PR, and
leaves `main` working. Nothing merges without tests passing.

## Risk register — read before starting

Three unknowns that can move the plan. All are probed in week 1 rather than discovered in
week 3.

| Risk | Impact | When we find out | Mitigation |
|---|---|---|---|
| **MPS throughput unknown** | Sets every token budget and the whole sweep schedule | M3, day 5 | Budgets in DESIGN §7 are provisional and get rewritten from measured tokens/sec. If throughput is bad, shrink `d_model` to 256 and the sweep budget to 25M tokens before shrinking the number of seeds. |
| **MPS operator gaps or numerical quirks** | Could force CPU (far too slow) or CUDA | M2, day 3 | Spike first (below). Fallback is a borrowed CUDA box for the flagship run; ablations stay local. |
| **Post-norm may diverge, not merely underperform** | A1's chart becomes a divergence story | M6 | Already designed for: a NaN run is recorded as `FAILED` with its last good loss. Divergence *is* the finding — the writeup frames it that way. |

**Spike, day 1 (throwaway, not merged):** 30 lines — build a tiny transformer on MPS, run
100 steps, print tokens/sec and peak memory. Answers the first two risks before any
architecture is committed. Lives in `spikes/`, gitignored, deleted after.

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

**Gate:** **measured tokens/sec and peak memory reported in the PR.** Token budgets in
DESIGN §7 get updated from real numbers here, and DESIGN.md is edited in the same PR.
This is the plan's main decision point.

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

## Blocked on sign-off

Five open questions in REQUIREMENTS §7. Three of them gate work:

- **Q1 ablation selection** — blocks M6, and partly M2, since the config fields have to
  cover the chosen conditions.
- **Q2 corpus** — blocks M1.
- **Q5 GitHub repo creation** — needs explicit approval before anything is pushed.

Q3 (W&B) and Q4 (repo name) can be decided later without stalling.
