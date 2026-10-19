#!/usr/bin/env bash
# Concise status of a run: per-stage latest state, job id, scheduler state, log path.
#   slurm/status.sh RUN_DIR
set -Eeuo pipefail
[[ $# -eq 1 && "$1" != -h && "$1" != --help ]] || { echo "usage: slurm/status.sh RUN_DIR" >&2; exit 2; }
RUN_DIR="$(cd "$1" && pwd)" || exit 2
[[ -f "$RUN_DIR/manifest.json" ]] || { echo "not a run dir (no manifest.json): $RUN_DIR" >&2; exit 2; }
R="$(dirname "${BASH_SOURCE[0]}")/lib/receipt.py"
printf '%-9s %-11s %-10s %-14s %s\n' STAGE STATE JOB SCHEDULER "TIME (UTC) / LOG"
for stage in setup prepare boost train eval train_sft eval_sft eval_base gpucheck; do  # states: SUBMITTED PREFLIGHT_OK STARTED SUCCEEDED FAILED
  f="$RUN_DIR/status/$stage.state"
  [[ -f "$f" ]] || continue
  state="$(python3 "$R" get "$f" state)"; job="$(python3 "$R" get "$f" job_id)"
  t="$(python3 "$R" get "$f" time_utc)"
  sched="-"
  if [[ -n "$job" && "$job" != "null" ]] && command -v squeue >/dev/null; then
    sched="$(squeue -h -j "$job" -o '%T' 2>/dev/null | head -n1 || true)"
    [[ -n "$sched" ]] || sched="$(sacct -j "$job" -X -n -P -o State,ExitCode 2>/dev/null | head -n1 | tr '|' ' ' || true)"
    [[ -n "$sched" ]] || sched="?"
  fi
  rec="$(ls -1t "$RUN_DIR"/receipts/"$stage"-*.json 2>/dev/null | head -n1 || true)"
  logp=""; [[ -n "$rec" ]] && logp="$(python3 "$R" get "$rec" log_out)"
  printf '%-9s %-11s %-10s %-14s %s  %s\n' "$stage" "$state" "${job:--}" "$sched" "$t" "$logp"
done
