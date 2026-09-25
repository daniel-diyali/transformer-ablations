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

## 2026-09-24 — M2 model

**Attention is a pure function, not a method.** `causal_attention(q, k, v)` sits outside
the module so the equivalence test can target it directly. That keeps the fast path out
of the model entirely — no config flag existing only for testing, and no chance a run
silently takes a different code path than the one under study. DESIGN §2.2 allowed a flag
routing to SDPA; this is strictly better and the flag was never added.

**Loss at init caught a real subtlety.** The first version of the test passed the input
as its own target and scored 4.26 against a ln(128)=4.85 baseline. The model was fine;
the test was asking it to predict the *current* token, which weight tying already solves
at initialisation — the residual stream carries the token embedding and the tied head
scores it against itself. RoPE showed the effect most strongly because it is the only
encoding that does not dilute the residual stream with an added position vector.

Worth remembering as an interview answer: weight tying plus a copying objective is a
shortcut, and the three positional encodings differ in how much they obscure it.

**Parameter count convention.** `num_params(non_embedding=True)` subtracts position
embeddings but not token embeddings, following nanoGPT, because weight tying means that
matrix is also the output head. An earlier DESIGN draft said 10.7M by subtracting the
token embedding too. Measured figure is 13.89M total, 13.79M non-embedding, and the
design now states the convention rather than just a number.

**pos_capacity settles an A2 question the plan left open.** A learned-encoding model
cannot run beyond its position table at all. Rather than error out and leave the
long-context arm with no data point, `pos_capacity` above `block_size` allocates
embeddings that training never reaches, so evaluation past the training context measures
exactly what untrained position embeddings do. That *is* the finding, and it is honest.

Note for A2's writeup: the three encodings cannot have identical parameter counts —
sinusoidal and RoPE have zero position parameters, learned has `n_positions * d_model`.
Unlike A1 and A3, A2's conditions are not parameter-matched, and saying so is better than
pretending otherwise.

### Next step

M3, the training loop, on `feat/training-loop`. Its gate compares real throughput against
the spike's 23,500 tok/s.

## 2026-09-24 — M3 training loop

**Gate passed, but the first measurement misled me.** End-to-end `train()` reported
20,634 tok/s against the spike's 23,500, a 12% gap. Two wrong hypotheses before the right
answer:

1. *`get_batch` is slow* — no. 1.0 ms out of a 348 ms step. Negligible.
2. *`loss.item()` forces a GPU sync every step* — plausible, and wrong. Measured with and
   without: 23,542 vs 23,526 tok/s. No cost on MPS at this size.

The isolated loop, including batch gathering, clipping and `.item()`, runs at
**23,542 tok/s** — the spike was accurate. The gap is entirely evaluation (4.73 s per
eval of 40 batches), checkpointing (0.11 s), and startup amortised over a short run. A
50M-token sweep run with default settings projects to **37.4 min**, inside the ceiling.

Lesson worth keeping: a cumulative `tokens/s` over a short run is dominated by fixed
startup cost. The 1.5M-token run looked *slower* than the 2M one for that reason alone.
Measure steady state, not averages over short runs.

**Resuming with a changed token budget is refused, and that is correct.** I wrote a test
assuming a run could be extended; it failed. The LR schedule is keyed to the budget, so a
run resumed under a different budget follows a curve matching neither. The config-equality
check catches it. Test now asserts the refusal and explains why.

**Evaluation uses a fixed seed** so every eval, in every run, scores identical validation
batches. A moving eval set would inject run-to-run noise indistinguishable from a real
difference between conditions — the one error this project cannot absorb.

**W&B is wired in behind `--wandb`** (Q3). Nothing downstream reads it; JSONL stays
canonical, so a missing account costs a run nothing but a dashboard.

### Next step

M4, baseline run, on `feat/baseline-run`: train the sweep config to its full 50M budget,
produce the first real loss curve and text samples. First artifact that is resume-legible
on its own.

## 2026-09-24 — M4 baseline run

**Baseline trained.** 50,003,968 tokens, 6,104 steps, validation loss **9.045 → 1.885**
in 26.7 minutes. Train and validation track each other the whole way — no overfitting,
expected when a run sees only 10.7% of the corpus.

**Throughput was higher than every benchmark: 31,202 tok/s** against the spike's 23,500
and M3's 23,542. Both earlier figures were measured while other work ran on the same
machine, so they were taken under contention. An uninterrupted half-hour run is the
number the sweep will actually experience. Budgets revised: 27-run sweep ~12 h, flagship
~1.8 h, project ~14 h rather than 19.

Worth remembering: benchmark on an idle machine, or say plainly that the number is a
floor. I reported 23,542 as though it were the loop's capability when it was the
loop's capability *under load*.

**Samples are genuinely coherent.** Grammatical, with narrative structure and characters
that persist across paragraphs — and a bird named Bunny that becomes a Bear a few lines
later. That failure mode is exactly what 13.9M parameters should produce, and it is more
honest to show it than to cherry-pick.

**Byte-level BPE has a hard floor of 257.** 256 byte values plus `<|endoftext|>`. A test
fixture asked for 256 and failed after training a tokenizer. `build_tokenizer` now
rejects it at the start and explains the floor.

**Log y-axis on loss curves, by default.** Loss falls from ~9 to under 2, and on a linear
axis the first plunge swallows the vertical space, squashing the plateau — where the
differences between ablation conditions actually live — into a few pixels.

### Next step

M5, the sweep runner, on `feat/sweep-runner`: `Study` definitions, expansion over
conditions × seeds, per-run failure isolation, `results.jsonl` aggregation.

## 2026-09-24 — M5 sweep runner

**The most important line in `experiments.py` is the unknown-override check.** A typo'd
condition key that was silently ignored would let all 27 runs complete with every
condition identical, and the resulting chart would read as a genuine finding of "no
difference". That failure is undetectable from the output, so it has to be impossible by
construction.

**Resume checks a config fingerprint, not just a name.** Skipping on name alone would let
a changed setting reuse stale results under the same study name, mixing two
configurations into one set of numbers.

**Failure isolation is per run.** A crash that halts the sweep is how a 27-run sweep
quietly becomes a 9-run sweep that nobody notices until the charts look thin.

**Tests now encode each study's own claim.** A1 and A3 are parameter-matched exactly
(13.793M in every condition); A2 is not and cannot be (learned 14.186M, sinusoidal and
RoPE 13.793M). Those are assertions about validity, not about code, and they are the only
thing that would ever catch the studies drifting into confoundedness.

**A short run of a full-size model is still a full-size model.** The CLI smoke test took
57 s with `--token-budget 512` because the architecture stayed at 13.9M parameters. Adding
model-size flags took the file from 82 s to 10 s. Worth remembering: shrink the model, not
just the budget.

**Two process slips worth recording.** First, I piped a CLI test through `tail`, which
masked its non-zero exit, and committed a broken CLI — amended after catching it. Second,
a `str.replace` on a test file silently did nothing because ruff had already reformatted
the anchor; the edit reported success and changed nothing. Third, while writing these very notes, I reused a
variable and wrote README content into PLAN.md — the script printed "README updated" and
had clobbered a different file. Restored from git and redone with an assertion on the
result, not just on the anchor.

All three are the same failure I write tests against: an operation that appears to
succeed while doing nothing, or doing something else. Assert on the anchor *and* on the
outcome, and check exit codes rather than piped output.

### Next step

M6, the ablations, on `feat/ablations`: run all 27, then build the per-study charts
showing individual seeds alongside the mean.
