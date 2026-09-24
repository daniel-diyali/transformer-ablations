# NOTES — transformer-ablations

Decisions and their reasons, newest last.

## 2026-09-24 — Project opened

**Why this project.** Chosen from a resume gap analysis against live Summer 2027 ML
intern postings. The gap it closes: PyTorch appears in the skills list but in no project;
the only modeling work on the resume is a sklearn logistic regression. Postings mention
PyTorch in 46 of ~120 ML intern listings, and every post-training role assumes a training
loop.

**Why ablations rather than just an implementation.** A from-scratch transformer is
commodity work — the public repos number in the thousands. Controlled experiments with
multiple seeds and honestly reported variance are not. The differentiator is the
experimental method, so the repo is named for it.

**Local MPS over Colab.** The M4 Pro has 24 GB of unified memory, which comfortably holds
a 14M-parameter model. The binding constraint on this project is the *number of
comparable runs*, not the speed of any one run, and free-tier session limits are hostile
to a 27-run sweep. Varying hardware between Colab sessions would also pollute timing
comparisons. Code stays device-agnostic (`cuda|mps|cpu`) so a longer flagship run could
burst to a borrowed GPU later.

**Ablation dimensions as config flags, not code forks.** The load-bearing design
decision. One model class covers every condition, so conditions cannot silently diverge
in some unrelated way, and one test suite covers them all.

**Three seeds per condition, minimum.** Most public ablation repos run one seed and
report noise as signal. Charts here plot individual seeds alongside the mean; if seeds
disagree about which condition wins, the chart has to show it. Seed count is the last
thing to cut if time runs short.

**Token budgets, not epochs.** Conditions with different batch shapes must still see
identical data volume or the comparison means nothing.

**Tokenization uses a library.** "From scratch" is a claim about the model and the
training loop. Hand-rolling BPE would cost days on the part of the project no
interviewer asks about.

**Python 3.11 via `uv`.** System Python here is 3.9.6, too old for current torch. `uv`
was already installed.

## 2026-09-24 — Signed off

Daniel reviewed the three docs and resolved all five open questions in one pass; every
recommendation was accepted. Ablations are A1 (norm placement) + A2 (positional encoding)
+ A3 (head count), corpus is TinyStories, W&B goes in behind an optional flag with JSONL
as the source of truth, repo name stays `transformer-ablations`, and the repo is public
from the first commit.

Consequence worth recording: because the repo is public from commit one, the git history
is part of the artifact. Branch names, commit messages, and PR descriptions are readable
by anyone evaluating the project, so they get the same care as the code.

## 2026-09-24 — Day-1 spike: both compute risks closed

Ran the throwaway spike before touching any architecture. torch 2.14.0, MPS live.

**Throughput: 23,500 tok/s** at the planned config (d384/L6/H6, batch 32, fp32, explicit
attention), 349 ms/step, 3.5 GB peak against 24 GB available.

The consequence that matters: the 50M-token sweep budget in DESIGN §7 was an estimate,
and the measurement says it holds. 35 minutes per run, under the 45-minute ceiling.
Full 27-run sweep ~16 h, flagship 2.4 h, ~18.5 h total — two or three overnight sessions.
No need to shrink `d_model`, and the `d256` fallback config is dead.

**No MPS operator gaps.** All ten probed ops passed, including complex64 multiply (so the
complex RoPE formulation is available, not just the real-valued one) and both bf16 and
fp16 autocast. No CUDA fallback needed for any milestone.

Two incidental findings worth keeping:

- **Batch 32 is the setting.** Batch 64 gave 23,900 vs 23,500 tok/s — 2% more throughput
  for 68% more memory. The device is already saturated at 32.
- **A3 is cheap.** The 12-head condition runs only 10% slower than 6 heads, so the head
  count study costs almost nothing beyond its run count.

Not measured: whether bf16 autocast actually beats fp32 here. It runs, but MPS gains are
often modest. Logged as an optional M3 optimization, never to be switched on mid-sweep
where it could confound a comparison.

The spike validates **speed only** — it trains on one fixed random batch, and its loss
climbing to ~30 is meaningless noise rather than a signal. Correctness is M2's job. Worth
stating plainly so nobody later mistakes a throughput spike for a working model.

### Next step

M0 scaffold on `chore/scaffold`: pyproject with pinned deps, ruff, pytest, CI. Committed
in parts per the new version-control rule.

## 2026-09-24 — M1 data pipeline

**`datasets` was dropped.** `load_dataset(..., streaming=True)` hung indefinitely on
TinyStories while the Hub API itself answered in 150 ms, so the dependency was buying a
failure rather than a convenience. Replaced with `huggingface_hub.hf_hub_download` plus
`pyarrow` reading the parquet shards directly — a smaller surface, and both libraries
were already installed as transitive dependencies of `datasets`.

**The split comes from upstream.** TinyStories ships its own train/validation split, so
FR2.3's determinism requirement is satisfied by construction. No seed to fix, and no way
for a reshuffle to silently change what validation loss means between runs. Simpler than
the seeded document split the design originally described, and strictly safer.

**Byte-level BPE, not word-level.** Every input is representable, so `decode(encode(x))`
holds for arbitrary text rather than only for text the vocabulary happens to cover.
Tested against characters absent from the training corpus.

**Corpus built.** 466,768,869 train tokens, 4,691,376 validation tokens at vocab 8,192,
from four shards in 2m15s — far faster than expected, since `tokenizers` is Rust and ran
at 522% CPU. 4.12 chars/token. A 50M sweep run sees 10.7% of the corpus, the 200M
flagship 42.8%, so no run repeats data. 890 MB of `.bin`, all gitignored.

**Ruff earned its place.** It caught two unescaped dots in `pytest.raises(match=...)`
patterns, which would have matched more loosely than intended and could have let a wrong
error message pass a test.

### Next step

M2, the model, on `feat/gpt-model`. The correctness milestone: attention must match
`F.scaled_dot_product_attention` numerically, and causality is tested by perturbation
rather than by inspecting the mask.
