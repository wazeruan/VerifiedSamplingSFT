#!/usr/bin/env bash
# Network-dependent preparation for clusters whose compute nodes have no internet:
# build the venv and download datasets + model weights, then record setup/prepare as
# SUCCEEDED in a fresh (or resumed) run so `submit.sh pipeline --stages boost,train,eval`
# can run fully offline.
#
# Only run this on a host where your site policy permits downloads and environment builds
# (e.g. a data-transfer node, or a login node if explicitly allowed). It does no GPU work.
#
#   slurm/fetch_online.sh --i-confirm-site-policy-allows-this \
#       (--run-dir runs/NAME [--config configs/a100/math_qwen3b.yaml] | --resume RUN_DIR) \
#       [--venv PATH] [--cache-root PATH] [--hf-home PATH] [--datasets "math math500 gsm8k"]
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
usage() { sed -n '2,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 2; }

CONFIRM=0 RUN_BASE="" RESUME="" CONFIG="configs/a100/math_qwen3b.yaml" VENV="" CACHE_ROOT=""
HF_HOME_DIR="" DATASETS="math math500 gsm8k"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --i-confirm-site-policy-allows-this) CONFIRM=1; shift ;;
    --run-dir) [[ -n "${2:-}" ]] || die "--run-dir needs a value"; RUN_BASE="$2"; shift 2 ;;
    --resume) [[ -n "${2:-}" ]] || die "--resume needs a value"; RESUME="$2"; shift 2 ;;
    --config) [[ -n "${2:-}" ]] || die "--config needs a value"; CONFIG="$2"; shift 2 ;;
    --venv) [[ -n "${2:-}" ]] || die "--venv needs a value"; VENV="$2"; shift 2 ;;
    --cache-root) [[ -n "${2:-}" ]] || die "--cache-root needs a value"; CACHE_ROOT="$2"; shift 2 ;;
    --hf-home) [[ -n "${2:-}" ]] || die "--hf-home needs a value"; HF_HOME_DIR="$2"; shift 2 ;;
    --datasets) [[ -n "${2:-}" ]] || die "--datasets needs a value"; DATASETS="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
done
[[ "$CONFIRM" -eq 1 ]] || die "pass --i-confirm-site-policy-allows-this (downloads/env builds on this host must be permitted)"
[[ -n "$RUN_BASE" || -n "$RESUME" ]] || die "pass --run-dir or --resume"
[[ -z "$RUN_BASE" || -z "$RESUME" ]] || die "--run-dir and --resume are mutually exclusive"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then echo "note: running inside job $SLURM_JOB_ID" >&2; fi
command -v uv >/dev/null || die "uv not on PATH"
for d in $DATASETS; do case "$d" in math|math500|gsm8k) ;; *) die "unknown dataset '$d'" ;; esac; done

# Reuse submit.sh's run-dir claim + code snapshot (dry submission of nothing is not
# possible), so claim and snapshot here with the same layout.
abspath() { python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(os.path.join(sys.argv[1], sys.argv[2]))))' "$PROJECT_DIR" "$1"; }
R="$PROJECT_DIR/slurm/lib/receipt.py"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
if [[ -n "$RESUME" ]]; then
  RUN_DIR="$(abspath "$RESUME")"
  [[ -f "$RUN_DIR/manifest.json" && -f "$RUN_DIR/code/uv.lock" ]] || die "not a run dir: $RUN_DIR"
else
  CONFIG="$(abspath "$CONFIG")"; [[ -f "$CONFIG" ]] || die "config not found: $CONFIG"
  base="$(abspath "$RUN_BASE")"; mkdir -p "$(dirname "$base")"; RUN_DIR=""
  for i in $(seq 0 999); do
    cand="$base"; [[ "$i" -gt 0 ]] && cand="$(printf '%s-%03d' "$base" "$i")"
    if mkdir "$cand" 2>/dev/null; then RUN_DIR="$cand"; break; fi
  done
  [[ -n "$RUN_DIR" ]] || die "could not claim a run dir under $base"
  mkdir -p "$RUN_DIR/code" "$RUN_DIR/status" "$RUN_DIR/logs" "$RUN_DIR/receipts" "$RUN_DIR/args"
  cp "$CONFIG" "$RUN_DIR/config.yaml"
  cp -R "$PROJECT_DIR/src" "$PROJECT_DIR/slurm" "$RUN_DIR/code/"
  cp "$PROJECT_DIR/pyproject.toml" "$PROJECT_DIR/uv.lock" "$RUN_DIR/code/"
  find "$RUN_DIR/code" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  chmod -R a-w "$RUN_DIR/code/src" "$RUN_DIR/code/slurm" 2>/dev/null || true
  python3 "$R" update "$RUN_DIR/manifest.json" "created_utc=$TS" "project_dir=$PROJECT_DIR" \
    "run_dir=$RUN_DIR" "code_dir=$RUN_DIR/code" "source_config=$CONFIG" "config=$RUN_DIR/config.yaml" \
    "git_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
fi
CODE="$RUN_DIR/code"
CACHE_ROOT="$(abspath "${CACHE_ROOT:-.cache}")"
if [[ -n "$VENV" ]]; then
  VENV="$(abspath "$VENV")"
else  # same lock-keyed default as submit.sh
  VENV="$CACHE_ROOT/venvs/$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest()[:12])' "$CODE/uv.lock")"
fi
export UV_PROJECT_ENVIRONMENT="$VENV" UV_CACHE_DIR="$CACHE_ROOT/uv" UV_LINK_MODE=copy
export HF_HOME="${HF_HOME_DIR:+$(abspath "$HF_HOME_DIR")}"; export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
export PYTHONPATH="$CODE/src"
LOG="$RUN_DIR/logs/fetch-$TS.log"
echo "run dir: $RUN_DIR  (log: $LOG)"
exec > >(tee -a "$LOG") 2>&1          # everything below is logged; errexit stays in force
export SLURM_JOB_ID="fetch-$TS"       # job id used in markers for this host-side "job"
CUR_STAGE=""
on_err() {
  local rc=$?
  [[ -n "$CUR_STAGE" ]] && python3 "$R" mark "$RUN_DIR" "$CUR_STAGE" FAILED "exit_code=$rc" "via=fetch_online" || true
  echo "FAILED during $CUR_STAGE (exit $rc); see $LOG" >&2
  exit "$rc"
}
trap on_err ERR
set -x

CUR_STAGE=setup
python3 "$R" mark "$RUN_DIR" setup STARTED "host=$(hostname)"
mkdir -p "$(dirname "$VENV")"
uv sync --project "$CODE" --frozen --extra vllm
"$VENV/bin/python" -c "import vsft, torch, transformers, vllm"   # verify before stamping
"$VENV/bin/python" -m vsft.hpc stamp-lock --lock "$CODE/uv.lock"
python3 "$R" mark "$RUN_DIR" setup SUCCEEDED "venv=$VENV" "via=fetch_online"

CUR_STAGE=prepare
python3 "$R" mark "$RUN_DIR" prepare STARTED "host=$(hostname)"
# shellcheck disable=SC2086  # dataset names are validated words
"$VENV/bin/python" -m vsft.cli prepare $DATASETS --out-dir "$RUN_DIR/data"
"$VENV/bin/python" -m vsft.hpc download-model --config "$RUN_DIR/config.yaml"
for f in "$RUN_DIR"/data/*.jsonl; do [[ -s "$f" ]] || { echo "empty dataset $f" >&2; false; }; done
python3 "$R" mark "$RUN_DIR" prepare SUCCEEDED "data_dir=$RUN_DIR/data" "via=fetch_online"
CUR_STAGE=""
set +x
cat <<EOF

Done. Now run the GPU stages offline:
  slurm/submit.sh pipeline --account ACCOUNT --resume "$RUN_DIR" --stages boost,train,eval \\
      --cache-root "$CACHE_ROOT"${HF_HOME_DIR:+ --hf-home "$HF_HOME"}
EOF
