# REQUIREMENTS — transformer-ablations

**Status:** signed off 2026-09-24
**Owner:** Daniel Diyali
**Drafted:** 2026-09-24

## 1. Purpose

Build a GPT-style decoder-only transformer from scratch in PyTorch, train it to produce
coherent text at small scale, and run controlled ablations that answer specific
architectural questions with measured, reproducible results.

The deliverable is not the model. The deliverable is **a set of findings backed by an
experiment harness that someone else could re-run.** The model is the instrument.

## 2. Why this project exists

Two goals, in priority order:

1. **Close the training gap.** The resume lists PyTorch but contains no project that
   trains a model. Every post-training and ML-engineer intern posting assumes a training
   loop. This project converts "uses LLM APIs" into "implements and trains transformers."
2. **Demonstrate experimental rigor.** Implementing a transformer is commodity work —
   thousands of public repos do it. Running controlled experiments with multiple seeds,
   reporting variance honestly, and drawing conclusions is not commodity work. That is
   the part that survives an interview.

Secondary: the implementation work overlaps the ARENA track already in progress, so it
advances the AI safety self-study rather than competing with it.

## 3. Functional requirements

### FR1 — Model implementation

- FR1.1 Decoder-only transformer written in PyTorch from primitive ops. Attention
  computed explicitly; `nn.Transformer`, `nn.MultiheadAttention`, and HuggingFace model
  classes are not permitted in the model path.
- FR1.2 Components: token embedding, positional encoding, N stacked blocks
  (causal multi-head self-attention + position-wise MLP, residual connections,
  layer norm), final layer norm, language-model head.
- FR1.3 Weight tying between token embedding and LM head, toggleable by config.
- FR1.4 Every architectural choice under ablation is a **config flag**, not a code fork.
  One model class covers all conditions.
- FR1.5 Autoregressive sampling with temperature and top-k.

### FR2 — Data pipeline

- FR2.1 Train a byte-pair-encoding tokenizer on the corpus. Vocabulary size is a config
  parameter.
- FR2.2 Pre-tokenize the corpus once into a flat `uint16` binary, memory-mapped at train
  time. No tokenization in the training loop.
- FR2.3 Deterministic train/validation split, fixed across all runs, so validation loss
  is comparable between conditions.
- FR2.4 Round-trip correctness: `decode(encode(text)) == text` for a sampled set of
  documents.

### FR3 — Training

- FR3.1 Full training loop written by hand: forward, loss, backward, optimizer step,
  gradient clipping, LR scheduling, gradient accumulation.
- FR3.2 AdamW with configurable betas, weight decay, and no-decay parameter groups
  (biases and layer-norm parameters excluded from weight decay).
- FR3.3 Runs are budgeted in **tokens processed**, not epochs or wall-clock, so that
  every ablation condition sees exactly the same amount of data.
- FR3.4 Periodic validation loss evaluation on a fixed held-out set.
- FR3.5 Checkpointing: save model, optimizer, and RNG state; resume must reproduce the
  loss curve of an uninterrupted run.
- FR3.6 Device-agnostic: MPS, CUDA, or CPU selected automatically, overridable by config.

### FR4 — Experiment harness

- FR4.1 A single run is fully described by a serializable config. The config is written
  into the run's output directory alongside results.
- FR4.2 **Every condition runs on at least 3 seeds.** Results report mean and spread
  across seeds. A single-seed difference is not a finding.
- FR4.3 Structured per-run logging: step, tokens seen, train loss, validation loss,
  learning rate, wall-clock, tokens/sec — appended as JSONL.
- FR4.4 A sweep runner executes a set of conditions × seeds and writes a machine-readable
  results file that the plotting layer consumes.
- FR4.5 Runs are resumable and idempotent: re-invoking a completed run does not redo it.
- FR4.6 Recorded provenance per run: git commit SHA, config hash, device, library
  versions.

### FR5 — Ablations

Three ablation studies, each stating a hypothesis before the run and reporting whether
the data supported it. FR5.1-5.3 are confirmed; FR5.4 is a stretch goal. Hypotheses and
reasoning are in DESIGN §6.

- FR5.1 Layer-norm placement: pre-norm vs post-norm.
- FR5.2 Positional encoding: learned vs sinusoidal vs rotary (RoPE), including behavior
  at evaluation contexts longer than the training context.
- FR5.3 Attention head count at fixed model width.
- FR5.4 (stretch) Learning-rate schedule: cosine with warmup vs constant.

### FR6 — Reporting

- FR6.1 Loss curves for the flagship run (train and validation on the same axes).
- FR6.2 One chart per ablation, showing seed variance, not just means.
- FR6.3 A findings writeup: hypothesis, method, result, and what it means — including
  results that contradicted the hypothesis.
- FR6.4 README covering setup, how to reproduce every figure, and hardware/time cost.
- FR6.5 Generated text samples from the flagship model.

## 4. Non-functional requirements

- **NFR1 Reproducibility.** Seeded runs on the same commit and device reproduce to within
  documented tolerance. Any nondeterminism is named in the README rather than hidden.
- **NFR2 Cost: $0.** Local hardware only (Apple M4 Pro, 24 GB unified memory). No paid
  compute, no paid API calls. If a run does not fit locally, the scope shrinks to fit.
- **NFR3 Runtime.** A single ablation run completes in ≤ 45 minutes so a full sweep
  finishes overnight. The flagship run may take longer.
- **NFR4 Readability.** The model file is written to be read. A reader who knows the
  transformer paper should be able to map equations to code without a guide.
- **NFR5 Tested.** Correctness of attention, masking, and the training loop is covered by
  tests that fail loudly when wrong. Tests run in under a minute on CPU.
- **NFR6 CI.** Lint and tests run on GitHub Actions for every pull request.
- **NFR7 Timeline.** Roughly 3 weeks part-time, finishing before the
  October–November internship deadline wave.

## 5. Non-goals

- Competing with any published model on quality. The scale is ~10–30M parameters.
- Distributed or multi-GPU training.
- Fine-tuning, RLHF, or instruction tuning — that is a separate, later project.
- A serving API, web UI, or deployment. This is a research repo.
- Novel architecture research. The ablations re-derive known results; the value is in
  the rigor of the derivation, not in discovering something new.
- Beating a well-tuned nanoGPT baseline on speed.

## 6. Assumptions

Each of these was a question in disguise. Resolved at sign-off; A2 later confirmed by
measurement rather than left as an assumption.

- **A1** Summer 2027 internship applications are the deadline driver, so shipping a
  complete, documented result in ~3 weeks outranks pushing the model's quality further.
- **A2** ~~Assumption~~ **confirmed by measurement 2026-09-24.** Local MPS runs at
  23,500 tok/s on the planned config, putting a sweep run at 35 minutes and the whole
  project at ~18.5 hours of overnight compute. No CUDA fallback needed. See DESIGN §5.1.
- **A3** TinyStories is the right corpus: small enough to train on quickly, simple enough
  that a ~14M-parameter model produces genuinely coherent English — which makes the
  qualitative samples compelling rather than embarrassing.
- **A4** Multiple seeds per condition matter more than more conditions. Three studies
  done rigorously beat six done once each.
- **A5** The repo will be public on GitHub under `daniel-diyali`, since its purpose is to
  be read by recruiters and interviewers.
- **A6** A tokenizer library (HuggingFace `tokenizers` or `sentencepiece`) is acceptable
  for BPE. "From scratch" applies to the model and training loop, not to tokenization.
- **A7** Weights & Biases free tier is acceptable as an *optional* logging backend. The
  JSONL logs remain the source of truth so the repo works with no account.

## 7. Decisions — resolved 2026-09-24

All five open questions answered by Daniel at sign-off. Nothing is blocking.

- **Q1 Ablations → A1 + A2 + A3.** Layer-norm placement, positional encoding (with
  long-context evaluation), and head count at fixed width. 9 conditions × 3 seeds =
  27 runs. A4 (LR schedule) stays a stretch goal, run only if the budget allows.
- **Q2 Corpus → TinyStories.** Coherent English at ~14M parameters matters more than a
  more personal corpus that would produce weaker samples at this scale.
- **Q3 Weights & Biases → yes, optional, behind a flag.** JSONL remains the source of
  truth; the repo must work fully with no account and no network.
- **Q4 Repo name → `transformer-ablations`.** Leads with the differentiator rather than
  the commodity part.
- **Q5 GitHub → create public immediately.** Work is off this machine from commit one,
  and the git history itself becomes evidence of how the project was built.

## 8. Definition of done

- [ ] Model trains end to end and generates coherent English samples.
- [ ] Three ablation studies complete, each with ≥3 seeds per condition.
- [ ] Charts for every study, showing variance.
- [ ] Findings writeup, including anything that contradicted the hypothesis.
- [ ] README lets a stranger reproduce every figure from a clean clone.
- [ ] CI green; tests cover attention, masking, and training-loop correctness.
- [ ] Public repo, clean history, every unit of work merged through a PR.
