#!/bin/bash
# Wait for the 27-run sweep to finish, then do the analysis that follows it.
#
# Everything here is unattended on purpose: the sweep ends in the small hours
# and the long-context evaluation plus the four figures take only minutes, so
# there is no reason for them to wait for someone to be awake.
#
#   scripts/finish_sweep.sh
#
# Writes progress to runs/finish.log. Safe to run more than once: the sweep
# skips completed runs and every figure is regenerated from JSONL.

set -u
cd "$(dirname "$0")/.." || exit 1

PY=.venv/bin/python
RESULTS=runs/results.jsonl
TARGET=27

echo "[$(date '+%H:%M:%S')] waiting for $TARGET runs"

# Poll rather than wait on a pid: the sweep may be restarted by a watchdog, so
# the count of recorded runs is the honest completion signal, not any one process.
while :; do
  done_count=$(wc -l < "$RESULTS" 2>/dev/null | tr -d ' ')
  [ "${done_count:-0}" -ge "$TARGET" ] && break
  sleep 120
done

echo "[$(date '+%H:%M:%S')] sweep complete ($done_count runs)"

echo "[$(date '+%H:%M:%S')] long-context evaluation (study A2)"
$PY -m minigpt.experiments context-eval || echo "  context-eval FAILED"

echo "[$(date '+%H:%M:%S')] figures"
for study in norm_placement pos_encoding head_count; do
  $PY -m minigpt.analysis study --study "$study" || echo "  $study chart FAILED"
done
$PY -m minigpt.analysis context-scaling || echo "  context-scaling chart FAILED"

echo "[$(date '+%H:%M:%S')] summary"
$PY -m minigpt.analysis table

echo "[$(date '+%H:%M:%S')] done — figures in figures/, nothing committed yet"
