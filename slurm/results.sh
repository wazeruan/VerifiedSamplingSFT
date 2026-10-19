#!/usr/bin/env bash
# Results of a run: boosting summary, training outcome, evaluation accuracies.
#   slurm/results.sh RUN_DIR
set -Eeuo pipefail
[[ $# -eq 1 && "$1" != -h && "$1" != --help ]] || { echo "usage: slurm/results.sh RUN_DIR" >&2; exit 2; }
RUN_DIR="$(cd "$1" && pwd)" || exit 2
python3 - "$RUN_DIR" <<'PY'
import glob, json, os, sys
run = sys.argv[1]
def load(p):
    try:
        with open(p) as f: return json.load(f)
    except (OSError, ValueError): return None
def state(stage):
    s = load(os.path.join(run, "status", f"{stage}.state")) or {}
    return s.get("state", "-")
print(f"run: {run}")
b = load(os.path.join(run, "boosted.summary.json"))
print(f"\n[boost] state={state('boost')}")
if b:
    print(f"  rows={b.get('num_rows')} accept_rate={b.get('accept_rate')} "
          f"mean_rounds={b.get('mean_rounds')} status={b.get('status')} "
          f"logprob_gain_vs_expert={b.get('mean_logprob_gain_vs_expert')}")
print(f"\n[train] state={state('train')}")
td = open(os.path.join(run, "status", "train.dir")).read().strip() if os.path.exists(os.path.join(run, "status", "train.dir")) else None
if td:
    r = load(os.path.join(td, "train_result.json"))
    print(f"  dir={td} final_step={r and r.get('final_step')} elapsed_s={r and r.get('elapsed_s')}")
    hist = os.path.join(td, "train_history.jsonl")
    if os.path.exists(hist):
        lines = [json.loads(l) for l in open(hist) if l.strip()]
        if lines: print(f"  last loss={lines[-1].get('loss')} at step {lines[-1].get('step')}")
print(f"\n[eval] state={state('eval')}")
for p in sorted(glob.glob(os.path.join(run, "eval", "*", "eval_summary.json"))):
    s = load(p) or {}
    accs = " ".join(f"{k}={v}" for k, v in (s.get("accuracy") or {}).items())
    print(f"  {os.path.basename(os.path.dirname(p))}: {accs}")
PY
