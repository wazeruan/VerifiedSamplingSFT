#!/usr/bin/env bash
# Show the newest log of a stage in a run (bounded).
#   slurm/logs.sh RUN_DIR [STAGE] [N_LINES]      (STAGE default: most recent submission)
set -Eeuo pipefail
[[ $# -ge 1 && "$1" != -h && "$1" != --help ]] || { echo "usage: slurm/logs.sh RUN_DIR [STAGE] [N]" >&2; exit 2; }
RUN_DIR="$(cd "$1" && pwd)" || exit 2
STAGE="${2:-}" N="${3:-200}"
[[ "$N" =~ ^[0-9]+$ ]] || { echo "N must be an integer" >&2; exit 2; }
pattern="${STAGE:+$STAGE-}*.out"
newest="$(ls -1t "$RUN_DIR"/logs/$pattern 2>/dev/null | head -n1 || true)"
[[ -n "$newest" ]] || { echo "no logs matching $RUN_DIR/logs/$pattern" >&2; exit 1; }
for f in "$newest" "${newest%.out}.err"; do
  [[ -f "$f" ]] || continue
  echo "===== $f (last $N lines)"
  tail -n "$N" "$f"
done
