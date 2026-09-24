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

### Next step

Day-1 throughput spike (throwaway, `spikes/`, not merged) to measure MPS tokens/sec and
peak memory and to surface any MPS operator gaps. Its numbers set the token budgets in
DESIGN §7, which currently hold estimates rather than measurements. Then M0 scaffold.
