# shellcheck shell=bash
# Sourced by every slurm/jobs/*.sbatch script, on the compute node, inside the allocation.
# Portable to bash >= 3.2 (no mapfile, no negative array indices, guarded empty arrays).
#
# Inputs (exported by slurm/submit.sh):
#   VSFT_PROJECT_DIR  checkout used for the shared venv (VSFT_VENV) and caches
#   VSFT_CODE_DIR     per-run code snapshot (src/, slurm/, uv.lock): what actually runs
#   VSFT_RUN_DIR VSFT_STAGE VSFT_RECEIPT VSFT_ARGS_FILE VSFT_CONFIG VSFT_VENV
#   VSFT_MODULES VSFT_CACHE_ROOT VSFT_HF_HOME VSFT_OFFLINE VSFT_GPU_TYPE VSFT_MIN_GPU_MEM_GB
#   VSFT_PREEMPT_GRACE (seconds the workload gets to checkpoint after USR1/TERM)
#
# Provides: vsft_init, vsft_probe_network URL, vsft_mark_started_cpu, vsft_preflight,
#           vsft_run CMD..., vsft_succeed key=value..., VSFT_PY, VSFT_STAGE_ARGS (array).

set -Eeuo pipefail

: "${VSFT_PROJECT_DIR:?VSFT_PROJECT_DIR not set (submit through slurm/submit.sh)}"
: "${VSFT_CODE_DIR:?VSFT_CODE_DIR not set}"
: "${VSFT_RUN_DIR:?VSFT_RUN_DIR not set}"
: "${VSFT_STAGE:?VSFT_STAGE not set}"
: "${VSFT_RECEIPT:?VSFT_RECEIPT not set}"

RECEIPT_PY="$VSFT_CODE_DIR/slurm/lib/receipt.py"
export VSFT_RECEIPT_PY="$RECEIPT_PY"
: "${VSFT_VENV:?VSFT_VENV not set}"
VSFT_PY="$VSFT_VENV/bin/python"
export VSFT_STOP_FILE="$VSFT_RUN_DIR/status/$VSFT_STAGE.${SLURM_JOB_ID:-nojob}.STOP"
VSFT_PHASE="init"
VSFT_CHILD=""
VSFT_DONE=0
VSFT_STAGE_ARGS=()

_log() { printf '[%s] [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$VSFT_STAGE" "$*"; }

_mark() {  # _mark STATE key=value...
  local state="$1"; shift
  python3 "$RECEIPT_PY" mark "$VSFT_RUN_DIR" "$VSFT_STAGE" "$state" "$@" || true
  python3 "$RECEIPT_PY" update "$VSFT_RECEIPT" "state=$state" "phase=$VSFT_PHASE" "$@" || true
}

_kill_child() {  # _kill_child SIGNAL : signal the workload's whole process group if possible
  local sig="$1"
  [[ -n "$VSFT_CHILD" ]] || return 0
  kill "-$sig" -- "-$VSFT_CHILD" 2>/dev/null || kill "-$sig" "$VSFT_CHILD" 2>/dev/null || true
}

_signal_lead() {  # must match each stage's #SBATCH --signal=B:USR1@<lead>
  case "$VSFT_STAGE" in boost|train) echo 600 ;; eval) echo 300 ;; setup|prepare) echo 120 ;; *) echo 60 ;; esac
}

_default_grace() {  # time to let the workload checkpoint; always < lead - 60 s
  local lead want cap
  lead="$(_signal_lead)"; cap=$((lead - 60)); [[ "$cap" -ge 10 ]] || cap=10
  case "$VSFT_STAGE" in boost|train) want=500 ;; eval) want=240 ;; *) want=60 ;; esac
  [[ -n "${VSFT_PREEMPT_GRACE:-}" ]] && want="$VSFT_PREEMPT_GRACE"
  if [[ "$want" -gt "$cap" ]]; then want="$cap"; fi
  echo "$want"
}

_stopped_record() {  # path of the workload's own "stopped resumably" record for this job
  case "$VSFT_STAGE" in
    train) [[ -n "${VSFT_TRAIN_DIR:-}" ]] && echo "$VSFT_TRAIN_DIR/STOPPED.json" ;;
    *) echo "$VSFT_RUN_DIR/status/$VSFT_STAGE.${SLURM_JOB_ID:-nojob}.STOPPED.json" ;;
  esac
}

_on_signal() {
  local sig="$1" grace waited=0 rc=143 reason="signal"
  grace="$(_default_grace)"   # per-stage cap so the trap finishes before the hard limit
  _log "received SIG$sig during phase=$VSFT_PHASE"
  if [[ -n "$VSFT_CHILD" ]] && kill -0 "$VSFT_CHILD" 2>/dev/null; then
    # Ask the workload to checkpoint and exit (it polls $VSFT_STOP_FILE between steps).
    : > "$VSFT_STOP_FILE"
    while kill -0 "$VSFT_CHILD" 2>/dev/null && [[ "$waited" -lt "$grace" ]]; do
      sleep 2; waited=$((waited + 2))
    done
    if kill -0 "$VSFT_CHILD" 2>/dev/null; then
      _log "workload did not stop within ${grace}s; sending TERM"
      _kill_child TERM; sleep 5; _kill_child KILL
    fi
    wait "$VSFT_CHILD" 2>/dev/null || true
    # Decide from the workload's own record (written with this job id), not the exit
    # code: a multi-GPU launcher turns the ranks' 75 into a generic failure code.
    local rec; rec="$(_stopped_record || true)"
    if [[ -n "$rec" && -f "$rec" && "$(python3 "$RECEIPT_PY" get "$rec" job_id)" == "${SLURM_JOB_ID:-}" ]]; then
      rc=75; reason="stopped_resumable"
    fi
  fi
  local hint=""
  if [[ "$reason" == "stopped_resumable" && "$VSFT_STAGE" == "train" ]]; then
    hint="$(python3 "$RECEIPT_PY" get "$VSFT_TRAIN_DIR/STOPPED.json" state_dir)"
    _log "resumable state: $hint (resubmit: slurm/submit.sh train --resume $VSFT_RUN_DIR --train-resume $hint ...)"
  elif [[ "$reason" == "stopped_resumable" ]]; then
    _log "partial output kept; resubmitting the same stage resumes it"
  fi
  _mark FAILED "signal=$sig" "exit_code=$rc" "reason=$reason" "resume_state=$hint"
  VSFT_DONE=1
  exit "$rc"
}

_on_exit() {
  local rc=$?
  if [[ "$VSFT_DONE" -eq 0 ]]; then
    if [[ "$rc" -eq 0 ]]; then
      rc=70  # script ended without vsft_succeed: treat as failure, never as success
      _log "job script exited without recording success"
    fi
    _mark FAILED "exit_code=$rc" "reason=exit"
    _log "FAILED in phase=$VSFT_PHASE with exit code $rc"
  fi
  exit "$rc"
}

trap '_on_signal TERM' TERM
trap '_on_signal INT' INT
trap '_on_signal USR1' USR1
trap _on_exit EXIT

vsft_init() {
  VSFT_PHASE="validate"
  _log "job ${SLURM_JOB_ID:-?} on $(hostname) run_dir=$VSFT_RUN_DIR code=$VSFT_CODE_DIR"
  [[ -n "${SLURM_JOB_ID:-}" ]] || { _log "not inside a Slurm allocation"; exit 64; }
  [[ -d "$VSFT_RUN_DIR" && -w "$VSFT_RUN_DIR" ]] || { _log "run dir not writable"; exit 65; }
  [[ -f "$VSFT_CODE_DIR/uv.lock" && -d "$VSFT_CODE_DIR/src/vsft" ]] || {
    _log "code snapshot missing at $VSFT_CODE_DIR"; exit 65; }
  python3 "$RECEIPT_PY" update "$VSFT_RECEIPT" "state=RUNNING" "phase=validate" \
    "node=$(hostname)" "job_id=${SLURM_JOB_ID}" || true

  if [[ -n "${VSFT_MODULES:-}" ]]; then
    command -v module >/dev/null 2>&1 || { _log "module command unavailable"; exit 66; }
    local m
    for m in $VSFT_MODULES; do
      set +u; module load "$m"; local mrc=$?; set -u   # Lmod init may touch unset vars
      [[ "$mrc" -eq 0 ]] || { _log "module load $m failed"; exit 66; }
    done
  fi

  local cache="${VSFT_CACHE_ROOT:-$VSFT_PROJECT_DIR/.cache}"
  export HF_HOME="${VSFT_HF_HOME:-$cache/huggingface}"
  export UV_CACHE_DIR="$cache/uv" VLLM_CACHE_ROOT="$cache/vllm" TRITON_CACHE_DIR="$cache/triton"
  export XDG_CACHE_HOME="$cache/xdg" TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
  export UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT="$VSFT_VENV"
  export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
  # Run the per-run code snapshot, not whatever the shared venv was last built from.
  export PYTHONPATH="$VSFT_CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
  if [[ "${VSFT_OFFLINE:-1}" == "1" ]]; then
    export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  fi
  mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME"

  # Extra vsft args: NUL-separated (keeps empty args and newlines intact).
  VSFT_STAGE_ARGS=()
  if [[ -n "${VSFT_ARGS_FILE:-}" && -s "$VSFT_ARGS_FILE" ]]; then
    local a
    while IFS= read -r -d '' a; do VSFT_STAGE_ARGS+=("$a"); done < "$VSFT_ARGS_FILE"
  fi
}

vsft_probe_network() {  # fail fast with a clear message when compute nodes have no internet
  local url="$1"
  VSFT_PHASE="network-probe"
  if ! python3 - "$url" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(urllib.request.Request(sys.argv[1], method="HEAD"), timeout=15)
except Exception as e:  # noqa: BLE001
    print(f"cannot reach {sys.argv[1]}: {e}", file=sys.stderr)
    sys.exit(1)
PY
  then
    _log "no internet access from this compute node. Run this stage where your site allows"
    _log "downloads (see README: slurm/fetch_online.sh), then submit --stages boost,train,eval."
    exit 69
  fi
}

vsft_mark_started_cpu() {  # CPU stages: the application is the download/sync itself
  _mark STARTED "node=$(hostname)"
}

vsft_preflight() {  # vsft_preflight [gpu] [--require-file F | --require-model M ...]
  VSFT_PHASE="preflight"
  local args=(--run-dir "$VSFT_RUN_DIR" --stage "$VSFT_STAGE" --require-lock "$VSFT_CODE_DIR/uv.lock")
  if [[ "${1:-}" == "gpu" ]]; then
    shift
    args+=(--require-gpu --require-bf16 --min-gpu-mem-gb "${VSFT_MIN_GPU_MEM_GB:-39}")
    if [[ -n "${VSFT_GPU_TYPE:-}" ]]; then args+=(--gpu-type "$VSFT_GPU_TYPE"); fi
    if [[ -n "${VSFT_GPU_CAPABILITY:-}" ]]; then args+=(--gpu-capability "$VSFT_GPU_CAPABILITY"); fi
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || true
  fi
  [[ -x "$VSFT_PY" ]] || { _log "project venv missing at $VSFT_VENV: run the setup stage"; exit 66; }
  "$VSFT_PY" -m vsft.hpc preflight "${args[@]}" "$@"   # writes PREFLIGHT_OK on success
  python3 "$RECEIPT_PY" update "$VSFT_RECEIPT" "state=PREFLIGHT_OK" "phase=preflight" || true
}

vsft_run() {  # run the workload in its own process group, in the background, so traps fire
  VSFT_PHASE="run"
  _log "exec: $*"
  export VSFT_STOPPED_RECORD; VSFT_STOPPED_RECORD="$(_stopped_record || true)"
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" &
  else
    "$@" &
  fi
  VSFT_CHILD=$!
  local rc=0
  wait "$VSFT_CHILD" || rc=$?
  VSFT_CHILD=""
  if [[ "$rc" -ne 0 ]]; then
    _log "workload exited with $rc"
    exit "$rc"
  fi
}

vsft_succeed() {
  VSFT_PHASE="done"
  _mark SUCCEEDED "exit_code=0" "$@"
  VSFT_DONE=1
  _log "SUCCEEDED"
}
