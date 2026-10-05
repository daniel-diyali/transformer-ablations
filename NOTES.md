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

## 2026-09-26 — thermal profiles, and two ways a long run dies quietly

The sweep pinned the GPU at 98-99% for hours on a laptop, so runs are now paced:
identical arithmetic with short idle gaps. `minigpt/thermal.py` explains why duty cycling
was chosen over smaller batches or gradient accumulation — those would cut memory but
would change results and invalidate the three runs already recorded at batch 32.

**A default that is right for a sweep can be wrong for a test.** The CLI defaults to the
`cool` profile, which idles 60s between runs. The CLI smoke test invoked `run` without a
profile, inherited that default, and sat in `sleep()` — the whole suite looked hung at 40%
with the process at 0% CPU and no output. It was not hung; it was obeying me. Anything on
a test's path now opts out of pacing explicitly. Worth generalising: when adding a default
that spends wall-clock, check what else picks it up.

**Two claims were verified by hand and then left unguarded.** The commit message recorded
that paced and unpaced runs produced identical validation loss to the last digit, but no
test referenced `thermal` at all, while the module docstring said the invariance was
"asserted directly in the tests". A verified claim and a guarded claim are not the same
thing — the first is true today, the second stays true. `tests/test_thermal.py` now asserts
both the bit-identical result and that a paced rerun still skips runs finished at full
speed.

**A backgrounded run dies with the shell that started it.** The sweep was launched with
`nohup ... &` from a tool session and was gone within the hour, having logged a `resuming`
line and nothing after it — the same way a backgrounded test run vanished mid-suite
earlier. `nohup` ignores SIGHUP but does not survive the process group being cleaned up.
Long runs are now started in their own session, and the check is `ps -o ppid` showing 1,
not merely that a pid exists:

    python -c 'import os,sys; os.fork() and sys.exit(0); os.setsid(); os.execv(...)'

macOS has no `setsid(1)`, hence the inline fork. Note also that `pgrep -f
'minigpt.experiments run'` matches the launching shell's own command line, so a liveness
guard built on it reports a sweep that does not exist. Match the interpreter process, or
check `ppid`.

### Next step

Still M6: the 27-run sweep is resuming under the `cool` profile at run 4 of 27. When it
finishes, run `context-eval`, build the three study charts plus the context-scaling chart,
and write `FINDINGS.md` reporting each hypothesis against what happened.

## 2026-09-26 — Thermal pacing, and three quiet deaths

**Daniel's machine was overheating.** Jarvis suspended the sweep mid-run; Daniel's
call was to rerun it cooler. New standing rule, now in AGENTS.md: ask before any GPU
or multi-hour job, and state duration and thermal impact up front.

**Duty cycling was chosen because it is scientifically inert.** Same batch, same data
order, same gradients — only wall-clock changes. Verified rather than assumed: same
seed, paced and unpaced, identical final loss to the last digit. Smaller batch or
gradient accumulation would cut memory but change results, and would have invalidated
the three runs already recorded at batch 32.

Consequence: `ThermalProfile` is deliberately **not** in `TrainConfig` and not in the
config fingerprint. Had it been, resuming a paced sweep would have refused to skip
those three runs and repeated a night of compute for no scientific reason.

**A lever that fires is not a lever that works.** The first `cool` profile paused 0.6s
every 4 steps. `paused_s` proved the sleeps happened exactly on schedule — and GPU
utilization stayed at 97–99%, identical to full speed. Sub-second gaps hold the duty
cycle but never let the GPU downclock. The same 63% duty as 6s pauses every 40 steps
drops utilization to 0–7% during each gap. Measured mean fell 98–99% → **78.1%**.

Second time in two days I've claimed an intervention worked without measuring the
thing I cared about (the first was blaming OneDrive for a GPU-bound slowdown).

**Three deaths, none the sweep's fault.** One reboot; twice the process was reaped
because `nohup cmd &` leaves the child in the launching shell's process group. macOS
has no `setsid(1)`. `scripts/launch_detached.py` uses `start_new_session=True`; the
sweep now runs with ppid 1. Every death was survivable only because runs checkpoint
and resume — that design has now paid for itself three times.

**Not verified:** actual temperature and fan speed. Reading them needs `powermetrics`
under sudo. Duty cycle and utilization are measured; the thermal outcome is inferred.

## 2026-09-26 — Holding on battery

Jarvis flagged that the laptop was unplugged at 95% with ~19 h of work left. Under
sustained GPU load that battery lasts three or four hours, so the sweep would have
flattened the machine and gone down with it — a fourth death, and the first that
would also have cost Daniel his laptop mid-afternoon.

The paced profiles now hold while unplugged and resume on power. Same reasoning as
every other lever here: waiting delays arithmetic without changing it, so a run
interrupted by an unplugged laptop produces the result it would have produced plugged
in. Counted into `paused_s` so throughput accounting stays honest.

**Detection fails open on purpose.** No `pmset`, a timeout, a non-macOS host — all
report mains power. A probe that cannot answer must not be able to hang a sweep.

Verified live: restarted on battery, skipped the four finished runs, and held before
run 5 with the GPU at 5%.

### Early A1 signal, 4 of 27 runs in

pre-norm 1.8851 / 1.8835 / 1.8867 (spread 0.0032); post-norm s0 1.8895.

The gap between conditions (~0.004) is about the size of the seed spread. If that
holds across the remaining post-norm seeds, A1's hypothesis — that post-norm ends at
*higher* loss — is only weakly supported at six layers, and the honest finding is
"indistinguishable from seed noise at this scale". Exactly the outcome three seeds
per condition exist to detect, and exactly the kind of result that gets quietly
rounded into a win in repos that report a bare mean.

## 2026-10-04 — All 27 runs in, and one condition that measured a bug

The sweep finished. Final means, three seeds each:

| study | condition | mean | spread |
|---|---|---|---|
| A1 norm_placement | pre | 1.8851 | 0.0033 |
| A1 norm_placement | post | 1.8926 | 0.0054 |
| A2 pos_encoding | learned | 1.8858 | 0.0041 |
| A2 pos_encoding | sinusoidal | 2.6397 | 0.1585 |
| A2 pos_encoding | rope | 1.8297 | 0.0094 |
| A3 head_count | h1 | 1.9146 | 0.0080 |
| A3 head_count | h3 | 1.8870 | 0.0067 |
| A3 head_count | h6 | 1.8851 | 0.0033 |
| A3 head_count | h12 | 1.8866 | 0.0019 |

**The early A1 read was wrong, and that is worth recording.** Four runs in I wrote that
the pre/post gap was "about the size of the seed spread" and would likely land on
"indistinguishable from seed noise". With all six runs the condition ranges are disjoint
— pre [1.8835, 1.8867], post [1.8895, 1.8949] — so post-norm is reliably worse. The
effect is small (0.0075 nats, roughly twice the spread) but it is an effect. Reading a
trend off a third of a study was premature in the direction of caution rather than hype,
which is the better way to be wrong, but it was still wrong.

**The analysis that follows the sweep had never been run as a script.** `finish_sweep.sh`
fired at 16:49 and every single command inside it died with `NameError`. Both
`experiments.py` and `analysis.py` had `if __name__ == "__main__": raise SystemExit(main())`
sitting in the middle of the file, with the functions `main()` dispatches to defined
below it. Imported — which is all the test suite ever did — that is harmless. Run as
`python -m`, the module stops executing at the exit, so `load_sweep`, `plot_study` and
`plot_context_scaling` did not exist yet when `main()` reached for them.

Nothing was lost; the sweep itself was fine and all four figures regenerated from the
existing records once the ordering was fixed. But a module that imports cleanly can still
be entirely broken as a script, and the suite had no test that ran either entrypoint the
way the automation does.

**Sinusoidal did not lose by 0.75 nats; it was handicapped.** The raw table from the
paper has per-component RMS 1/sqrt(2) ≈ 0.707, while token embeddings initialise at std
0.02. Added straight onto the embedding, the position signal outweighed token identity
35x and carried 99.92% of the energy entering block one. The model spent the whole run
digging the token out from under it.

**The seed spread is what gave it away.** 0.1585 across three seeds, where every other
condition in the sweep sits between 0.0019 and 0.0094. A condition that is both a
large outlier in mean *and* a large outlier in variance is a bug far more often than it
is a finding. The mean alone would have read as a plausible result.

**Why the existing test missed it.** `test_loss_at_init` asserts loss at initialisation
is near ln(8192) ≈ 9.01, and a model with a 35x position signal still scores there,
because the final LayerNorm strips the overall scale before the tied head sees it.
Checking that a number is plausible is not checking that the thing producing it is sound.
Third time in this project that an intervention looked verified because the wrong
quantity was measured.

**The fix scales the table to the embedding's init std, not by sqrt(d_model).** The
paper's convention multiplies the embeddings up instead; applied here that would have
left sinusoidal at 76% positional share — the same confound, an order of magnitude
smaller. Matching `EMBED_INIT_STD` puts learned and sinusoidal at 50.00% and 49.88%, so
A2 compares trainability, which is the variable it is supposed to vary. The table's shape
is untouched, so the positional information it carries is identical; only amplitude
changes. Verified on a 2M-token spike before touching the sweep.

Six new assertions pin the properties the bug violated rather than the constant itself,
including the `t > n_positions` branch — the one A2's long-context arm runs through, and
the one where a raw table would have gone unnoticed because nothing else inspects its
amplitude.

**A2's headline survives intact**, because it never depended on sinusoidal. Validation
loss against evaluation context, trained at 256:

| context | learned | rope |
|---|---|---|
| 128 | 1.9708 | 1.9170 |
| 256 | 1.8858 | 1.8297 |
| 512 | 3.2019 | 2.0001 |
| 1024 | 4.0100 | 2.5769 |

Learned collapses past the training context; RoPE degrades gracefully. That is the
hypothesis confirmed, and it is the figure the study was chosen for.

### Blocked — three reruns need GPU time

`pos_encoding-sinusoidal-s0/s1/s2` have to be rerun on the fixed scale before A2 can be
reported in full. At the 19,982 tok/s the spike measured, 50M tokens is ~42 min a run:
**~2.1 h at full speed, ~3.3 h under `cool`**, plus a few minutes to redo the 12
sinusoidal rows in `context_eval.jsonl`. That is a multi-hour GPU job, so it waits for
Daniel per AGENTS.md. Reruns mean deleting the three run directories and their rows
first — the sweep skips anything already recorded.

### Next step

`FINDINGS.md` is written with A1, A3 and A2's learned-vs-RoPE arm reported in full and
the sinusoidal arm marked pending, on `docs/findings`. After approval and reruns:
regenerate the two A2 charts, update both result tables, and replace the pending section.
