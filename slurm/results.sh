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
geos = {}
for stage, what in (("train", "Sampling SFT"), ("train_sft", "plain SFT baseline"), ("train_paper", "authors' method")):
    st = state(stage)
    dfile = os.path.join(run, "status", f"{stage}.dir")
    if st == "-" and not os.path.exists(dfile):
        continue
    print(f"\n[{stage}] {what}: state={st}")
    td = open(dfile).read().strip() if os.path.exists(dfile) else None
    if td:
        r = load(os.path.join(td, "train_result.json"))
        m = load(os.path.join(td, "train_manifest.json")) or {}
        g = m.get("geometry") or {}
        geos[stage] = (g.get("num_examples"), g.get("total_steps"))
        print(f"  dir={td} final_step={r and r.get('final_step')} elapsed_s={r and r.get('elapsed_s')}")
        print(f"  examples={g.get('num_examples')} total_steps={g.get('total_steps')} data={m.get('data')}")
        hist = os.path.join(td, "train_history.jsonl")
        if os.path.exists(hist):
            lines = []
            for l in open(hist):
                try:
                    lines.append(json.loads(l))
                except ValueError:
                    pass
            if lines: print(f"  last loss={lines[-1].get('loss')} at step {lines[-1].get('step')}")
if len(geos) > 1 and len(set(geos.values())) > 1:
    print(f"\nWARNING: the training arms used different example counts / step counts: {geos}")
print()
for stage in ("eval_base", "eval_sft", "eval", "eval_paper"):
    if state(stage) != "-":
        print(f"[{stage}] state={state(stage)}")
PY

# Base vs plain SFT vs Sampling SFT (stdlib-only helper from the run's code snapshot).
CMP="$RUN_DIR/code/src/vsft/compare.py"
[[ -f "$CMP" ]] || CMP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/src/vsft/compare.py"
if [[ -d "$RUN_DIR/eval" ]]; then
  echo
  python3 "$CMP" "$RUN_DIR/eval" || true
fi
