#!/usr/bin/env bash
# Noninteractive Slurm submission wrapper for VerifiedSamplingSFT (H100 defaults).
# Run on the cluster login node from a checkout of this repository. See --help.
# Portable to bash >= 3.2.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: slurm/submit.sh STAGE --account ACCOUNT [options] [-- extra vsft args]

STAGE
  setup      CPU   build the shared venv from the run's code snapshot (needs internet)
  prepare    CPU   download MATH/MATH-500/GSM8K + model weights (needs internet)
  boost      GPU   verifier-guided boosting  -> RUN_DIR/boosted.jsonl
  train      GPU   Sampling SFT on boosted traces       -> RUN_DIR/sft[-NNN]/final
  eval       GPU   accuracy of the Sampling SFT model   -> RUN_DIR/eval/sampling-JOBID
  train_sft  GPU   plain SFT baseline: raw expert traces of the same boosted rows -> RUN_DIR/sft-baseline
  eval_sft   GPU   accuracy of the plain SFT model      -> RUN_DIR/eval/sft-JOBID
  eval_base  GPU   accuracy of the initial (untrained) model -> RUN_DIR/eval/base-JOBID
  paper_sample GPU the authors' method: block-wise MCMC sampling (paper Algorithm 1)
                   over the same training rows -> RUN_DIR/paper_sampled.jsonl
  train_paper GPU  SFT on the authors'-method samples (same config)  -> RUN_DIR/sft-paper
  eval_paper GPU   accuracy of the authors'-method model              -> RUN_DIR/eval/paper-JOBID
  gpucheck   GPU   allocation/accelerator/environment check only
  paper            the authors' method alone, from scratch, in its own run directory
                   (runs/paper-<config>): setup, prepare, paper_sample, train_paper,
                   eval_paper, chained afterok. Same dataset, test sets and training
                   settings as the other arms. Example:
                     slurm/submit.sh paper --account ACCOUNT --min-gpu-mem-gb 79
  pipeline         submit --stages (default setup,prepare,boost,train,eval) chained afterok
                   (no internet on compute nodes? run slurm/fetch_online.sh where your site
                   allows downloads, then: pipeline --stages boost,train,eval)
  compare          Sampling SFT vs plain SFT vs base, as a dependency graph:
                     setup -> prepare -> boost -> train     -> eval
                                               -> train_sft -> eval_sft
                                      -> eval_base
                   (default: all 8 stages; --stages picks a subset, e.g. after fetch_online:
                   --stages boost,train,eval,train_sft,eval_sft,eval_base).
                   Then: slurm/results.sh RUN_DIR prints the comparison table.

Required
  --account NAME          Slurm allocation account (forwarded to every sbatch)

Run directory (exactly one of)
  --run-dir PATH          fresh run: atomically claims PATH, PATH-001, ... (default
                          runs/<config name>); snapshots code + config into it
  --resume RUN_DIR        reuse an existing run (its code snapshot and config.yaml)

Resources (GPU stages; defaults suit one H100)
  --gpus TYPE:N           GPU stages, default h100:1 (boost/eval always use one GPU)
  --train-gpus TYPE:N     train/train_sft only (default: --gpus); N>1 -> accelerate launch,
                          e.g. h100:2 with --accelerate-config for FSDP full fine-tuning
  --cpus N                default 8
  --mem SIZE              host RAM per job, default 120G (not GPU memory)
  --time D-HH:MM:SS       default: per-stage #SBATCH value
  --partition NAME, --qos NAME, --reservation NAME, --constraint EXPR
  --cpu-partition NAME    partition for setup/prepare (default: --partition)
  --cpu-mem SIZE, --cpu-time T   overrides for CPU stages

Environment and storage
  --config PATH           vsft YAML (default configs/h100/math_qwen35.yaml), fresh runs only
  --module NAME           `module load NAME` inside each job (repeatable)
  --venv PATH             environment (default CACHE_ROOT/venvs/<uv.lock sha>: one per lock,
                          so runs with different locks never rebuild each other's venv)
  --cache-root PATH       HF/uv/vLLM/Triton caches (default PROJECT/.cache; use scratch)
  --hf-home PATH          override the HF cache only
  --online                allow Hub access from GPU stages (default: offline)
  --gpu-type STR          preflight requires this in the GPU name (default H100; '' = any)
  --min-gpu-mem-gb N      preflight minimum GPU memory (default 79: H100 80 GB)
  --gpu-capability X.Y    preflight requires this CUDA compute capability and matching torch
                          kernels (default 9.0 = H100 sm_90; '' = any)
  --preempt-grace SEC     time the workload gets to checkpoint after the USR1 warning
                          (default per stage: boost/train 500, eval 240, others 60; always
                          capped per stage at its --signal lead minus 60 s)
  --project-dir PATH      checkout to snapshot code from (default: this repository)

Stage inputs
  --stages LIST           pipeline stages, comma separated, in order
  --input PATH            boost input JSONL (default RUN_DIR/data/math_l345_train.jsonl)
  --train-data PATH       train input JSONL (default RUN_DIR/boosted.jsonl)
  --train-resume DIR      resume training from RUN_DIR/sft*/state-* (train stage only)
  --accelerate-config F   accelerate config for multi-GPU train (e.g. FSDP)
  --eval-model PATH|ID    evaluate this model instead of the run's trained model
  --eval-data PATH        eval JSONL (repeatable; default the run's MATH/MATH-500/GSM8K)
  --datasets "A B"        datasets for prepare (default "math math500 gsm8k")
  --depend-on JOBID[:..]  afterok dependency for the first submitted stage (reconciled)
  --allow-duplicate       submit even if the same stage of this run is queued/running
  --refresh-code          with --resume: replace the run's code snapshot with this checkout
                          (keeps config.yaml, data and outputs; the old snapshot is kept as
                          code.prev-<UTC>; refused while any job of the run is queued/running)

Submission behaviour
  --monitor-timeout SEC   bounded startup monitoring of the first job (default 300)
  --no-monitor            skip startup monitoring (outcome UNVERIFIED, exit 3)
  --min-free-gb N         required free disk at run dir and cache root (default 50)
  --dry-run               validate and print the sbatch commands; submit nothing
  -h, --help

Extra arguments after `--` go to the vsft command of a single stage, e.g.
  -- --set boost.num_candidates=8 --limit 100

Outcome (exit code): STARTED (0) | PENDING (10) | FAILED (1) | UNKNOWN/UNVERIFIED (3)
| SUBMISSION_REJECTED (4) | invalid usage/validation (2)
STARTED means the application initialized (model/engine loaded) inside the allocation.
EOF
}

SUBMITTED_IDS=()
log() { printf '[submit] %s\n' "$*" >&2; }
die() {
  echo "error: $*" >&2
  if [[ ${#SUBMITTED_IDS[@]} -gt 0 ]]; then
    echo "note: already submitted for this run: ${SUBMITTED_IDS[*]} (run dir $RUN_DIR)" >&2
  fi
  exit 2
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
case "$1" in -h|--help) usage; exit 0 ;; esac
STAGE="$1"; shift
case "$STAGE" in setup|prepare|boost|train|eval|train_sft|eval_sft|eval_base|paper_sample|train_paper|eval_paper|gpucheck|pipeline|paper|compare) ;; *) die "unknown stage '$STAGE' (see --help)" ;; esac

ACCOUNT="" RUN_BASE="" RESUME="" CONFIG="" CONFIG_GIVEN=0 PROJECT_DIR="$DEFAULT_PROJECT_DIR"
GPUS="h100:1" TRAIN_GPUS="" CPUS="8" MEM="120G" TIME="" PARTITION="" QOS="" RESERVATION="" CONSTRAINT=""
CPU_PARTITION="" CPU_MEM="" CPU_TIME="" MODULES=() VENV="" CACHE_ROOT="" HF_HOME_DIR=""
OFFLINE=1 GPU_TYPE="H100" GPU_CAPABILITY="9.0" MIN_GPU_MEM_GB="79" PREEMPT_GRACE="" STAGES_OPT="" INPUT=""
TRAIN_DATA="" TRAIN_RESUME="" ACC_CONFIG="" EVAL_MODEL="" EVAL_DATA=() DATASETS=""
DEPEND_ON="" ALLOW_DUP=0 REFRESH_CODE=0 MONITOR_TIMEOUT=300 NO_MONITOR=0 MIN_FREE_GB=50 DRY_RUN=0 EXTRA=()
RUN_DIR=""

need() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 requires a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) need "$@"; ACCOUNT="$2"; shift 2 ;;
    --run-dir) need "$@"; RUN_BASE="$2"; shift 2 ;;
    --resume) need "$@"; RESUME="$2"; shift 2 ;;
    --config) need "$@"; CONFIG="$2"; CONFIG_GIVEN=1; shift 2 ;;
    --project-dir) need "$@"; PROJECT_DIR="$2"; shift 2 ;;
    --gpus) need "$@"; GPUS="$2"; shift 2 ;;
    --train-gpus) need "$@"; TRAIN_GPUS="$2"; shift 2 ;;
    --cpus) need "$@"; CPUS="$2"; shift 2 ;;
    --mem) need "$@"; MEM="$2"; shift 2 ;;
    --time) need "$@"; TIME="$2"; shift 2 ;;
    --partition) need "$@"; PARTITION="$2"; shift 2 ;;
    --qos) need "$@"; QOS="$2"; shift 2 ;;
    --reservation) need "$@"; RESERVATION="$2"; shift 2 ;;
    --constraint) need "$@"; CONSTRAINT="$2"; shift 2 ;;
    --cpu-partition) need "$@"; CPU_PARTITION="$2"; shift 2 ;;
    --cpu-mem) need "$@"; CPU_MEM="$2"; shift 2 ;;
    --cpu-time) need "$@"; CPU_TIME="$2"; shift 2 ;;
    --module) need "$@"; MODULES+=("$2"); shift 2 ;;
    --venv) need "$@"; VENV="$2"; shift 2 ;;
    --cache-root) need "$@"; CACHE_ROOT="$2"; shift 2 ;;
    --hf-home) need "$@"; HF_HOME_DIR="$2"; shift 2 ;;
    --online) OFFLINE=0; shift ;;
    --offline) OFFLINE=1; shift ;;   # default; kept for compatibility
    --gpu-type) [[ $# -ge 2 ]] || die "--gpu-type requires a value"; GPU_TYPE="$2"; shift 2 ;;
    --min-gpu-mem-gb) need "$@"; MIN_GPU_MEM_GB="$2"; shift 2 ;;
    --gpu-capability) [[ $# -ge 2 ]] || die "--gpu-capability requires a value"; GPU_CAPABILITY="$2"; shift 2 ;;
    --preempt-grace) need "$@"; PREEMPT_GRACE="$2"; shift 2 ;;
    --stages) need "$@"; STAGES_OPT="$2"; shift 2 ;;
    --input) need "$@"; INPUT="$2"; shift 2 ;;
    --train-data) need "$@"; TRAIN_DATA="$2"; shift 2 ;;
    --train-resume) need "$@"; TRAIN_RESUME="$2"; shift 2 ;;
    --accelerate-config) need "$@"; ACC_CONFIG="$2"; shift 2 ;;
    --eval-model) need "$@"; EVAL_MODEL="$2"; shift 2 ;;
    --eval-data) need "$@"; EVAL_DATA+=("$2"); shift 2 ;;
    --datasets) need "$@"; DATASETS="$2"; shift 2 ;;
    --depend-on) need "$@"; DEPEND_ON="$2"; shift 2 ;;
    --allow-duplicate) ALLOW_DUP=1; shift ;;
    --refresh-code) REFRESH_CODE=1; shift ;;
    --monitor-timeout) need "$@"; MONITOR_TIMEOUT="$2"; shift 2 ;;
    --no-monitor) NO_MONITOR=1; shift ;;
    --min-free-gb) need "$@"; MIN_FREE_GB="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; EXTRA=("$@"); break ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
done

# ---------------------------------------------------------------- validation (cheap only)
[[ -n "$ACCOUNT" ]] || die "--account is required"
[[ "$ACCOUNT" =~ ^[A-Za-z0-9_.-]+$ ]] || die "--account has unexpected characters"
[[ -z "$RUN_BASE" || -z "$RESUME" ]] || die "--run-dir and --resume are mutually exclusive"
[[ "$REFRESH_CODE" -eq 0 || -n "$RESUME" ]] || die "--refresh-code needs --resume RUN_DIR"
[[ -z "$RESUME" || "$CONFIG_GIVEN" -eq 0 ]] || die "--config only applies to fresh runs (a resumed run keeps its config.yaml)"
[[ "$GPUS" =~ ^[A-Za-z0-9_]+:[1-8]$ ]] || die "--gpus must look like h100:1 .. h100:8"
[[ -z "$TRAIN_GPUS" || "$TRAIN_GPUS" =~ ^[A-Za-z0-9_]+:[1-8]$ ]] || die "--train-gpus must look like h100:2"
if [[ ( "$STAGE" == "pipeline" || "$STAGE" == "paper" || "$STAGE" == "compare" ) && "${GPUS##*:}" -gt 1 ]]; then
  # boost/eval run on one GPU; a multi-GPU --gpus would leave GPUs idle for hours.
  die "--gpus ${GPUS} would apply to every GPU stage; use --train-gpus for multi-GPU training"
fi
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"
[[ "$MEM" =~ ^[1-9][0-9]*[KMGT]?$ ]] || die "--mem must look like 120G"
[[ -z "$CPU_MEM" || "$CPU_MEM" =~ ^[1-9][0-9]*[KMGT]?$ ]] || die "--cpu-mem must look like 32G"
[[ "$MONITOR_TIMEOUT" =~ ^[0-9]+$ ]] || die "--monitor-timeout must be seconds"
[[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || die "--min-free-gb must be an integer"
[[ -z "$PREEMPT_GRACE" || "$PREEMPT_GRACE" =~ ^[0-9]+$ ]] || die "--preempt-grace must be seconds"
if [[ -n "$PREEMPT_GRACE" ]] && (( PREEMPT_GRACE >= 540 )); then
  die "--preempt-grace must be < 540 s (the longest stage warning is 600 s, minus margin)"
fi
[[ "$MIN_GPU_MEM_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--min-gpu-mem-gb must be a number"
[[ -z "$GPU_CAPABILITY" || "$GPU_CAPABILITY" =~ ^[0-9]+\.[0-9]$ ]] || die "--gpu-capability must look like 9.0"
[[ -z "$DEPEND_ON" || "$DEPEND_ON" =~ ^[0-9]+(:[0-9]+)*$ ]] || die "--depend-on must be JOBID[:JOBID...]"
[[ -z "$TRAIN_RESUME" || "$STAGE" == "train" || "$STAGE" == "train_sft" || "$STAGE" == "train_paper" ]] || die "--train-resume is only valid for the train, train_sft or train_paper stage"
[[ -z "$STAGES_OPT" || "$STAGE" == "pipeline" || "$STAGE" == "paper" || "$STAGE" == "compare" ]] || die "--stages is only valid with pipeline, paper or compare"
[[ ${#EXTRA[@]} -eq 0 || ( "$STAGE" != "pipeline" && "$STAGE" != "paper" && "$STAGE" != "compare" ) ]] || die "extra vsft args need a single stage"
if [[ "$STAGE" == "compare" ]]; then
  # One model/data override would apply to several branches and break the comparison.
  [[ -z "$EVAL_MODEL" && -z "$TRAIN_DATA" && -z "$TRAIN_RESUME" ]] \
    || die "--eval-model, --train-data and --train-resume need a single stage, not compare"
fi
if [[ -n "$DATASETS" ]]; then
  for d in $DATASETS; do case "$d" in math|math500|gsm8k) ;; *) die "unknown dataset '$d'" ;; esac; done
fi

PROJECT_DIR="$(cd "$PROJECT_DIR" 2>/dev/null && pwd)" || die "--project-dir does not exist"
[[ -f "$PROJECT_DIR/pyproject.toml" && -f "$PROJECT_DIR/uv.lock" && -d "$PROJECT_DIR/src/vsft" ]] \
  || die "$PROJECT_DIR is not a VerifiedSamplingSFT checkout"
command -v python3 >/dev/null || die "python3 is required on the submission host (receipts)"
RECEIPT_PY="$PROJECT_DIR/slurm/lib/receipt.py"
if [[ "$DRY_RUN" -eq 0 ]]; then
  for c in sbatch squeue sacct; do command -v "$c" >/dev/null || die "$c not found: run on a Slurm login node"; done
fi

abspath() { python3 -c 'import os,sys; print(os.path.abspath(os.path.expanduser(os.path.join(sys.argv[1], sys.argv[2]))))' "$PROJECT_DIR" "$1"; }

# Job script per stage: the comparison variants reuse the train/eval scripts.
script_of() {
  case "$1" in train_sft|train_paper) echo train ;; eval_sft|eval_base|eval_paper) echo eval ;; *) echo "$1" ;; esac
}
# Predecessor of each stage in the compare graph ("" = root).
compare_pred_of() {
  case "$1" in
    setup) echo "" ;; prepare) echo setup ;; boost) echo prepare ;; train) echo boost ;;
    eval) echo train ;; train_sft) echo boost ;; eval_sft) echo train_sft ;;
    eval_base) echo prepare ;; paper_sample) echo prepare ;; train_paper) echo paper_sample ;;
    eval_paper) echo train_paper ;; *) echo "" ;;
  esac
}
if [[ "$STAGE" == "pipeline" ]]; then
  IFS=',' read -r -a STAGES <<< "${STAGES_OPT:-setup,prepare,boost,train,eval}"
  [[ ${#STAGES[@]} -gt 0 ]] || die "--stages is empty"
  for s in "${STAGES[@]}"; do
    case "$s" in setup|prepare|boost|train|eval) ;; *) die "invalid pipeline stage '$s'" ;; esac
  done
elif [[ "$STAGE" == "paper" ]]; then
  # The authors' method on its own, from scratch: setup, prepare, paper sampling, SFT, eval.
  IFS=',' read -r -a STAGES <<< "${STAGES_OPT:-setup,prepare,paper_sample,train_paper,eval_paper}"
  [[ ${#STAGES[@]} -gt 0 ]] || die "--stages is empty"
  for s in "${STAGES[@]}"; do
    case "$s" in setup|prepare|paper_sample|train_paper|eval_paper) ;; *)
      die "invalid paper stage '$s' (paper runs setup, prepare, paper_sample, train_paper, eval_paper)" ;;
    esac
  done
elif [[ "$STAGE" == "compare" ]]; then
  IFS=',' read -r -a STAGES <<< "${STAGES_OPT:-setup,prepare,boost,train,eval,train_sft,eval_sft,eval_base,paper_sample,train_paper,eval_paper}"
  [[ ${#STAGES[@]} -gt 0 ]] || die "--stages is empty"
  for s in "${STAGES[@]}"; do
    case "$s" in setup|prepare|boost|train|eval|train_sft|eval_sft|eval_base|paper_sample|train_paper|eval_paper) ;; *) die "invalid compare stage '$s'" ;; esac
  done
else
  STAGES=("$STAGE")
fi
has_stage() { local s; for s in "${STAGES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }
if [[ "$STAGE" == "compare" ]]; then
  # Topological order check: a stage's predecessor must come before it when both are given.
  seen=" "
  for s in "${STAGES[@]}"; do
    p="$(compare_pred_of "$s")"
    if [[ -n "$p" ]] && has_stage "$p" && [[ "$seen" != *" $p "* ]]; then
      die "--stages: $p must come before $s"
    fi
    seen="$seen$s "
  done
fi
for s in "${STAGES[@]}"; do
  [[ -f "$PROJECT_DIR/slurm/jobs/$(script_of "$s").sbatch" ]] || die "missing job script for stage $s"
done
if has_stage setup && [[ ${#MODULES[@]} -eq 0 ]] && ! command -v uv >/dev/null; then
  die "setup needs uv on PATH in the job: install it (https://docs.astral.sh/uv/) or pass --module"
fi

CACHE_ROOT="$(abspath "${CACHE_ROOT:-.cache}")"
[[ -z "$VENV" ]] || VENV="$(abspath "$VENV")"
[[ -z "$HF_HOME_DIR" ]] || HF_HOME_DIR="$(abspath "$HF_HOME_DIR")"
[[ -z "$ACC_CONFIG" ]] || { ACC_CONFIG="$(abspath "$ACC_CONFIG")"; [[ -f "$ACC_CONFIG" ]] || die "accelerate config not found: $ACC_CONFIG"; }
[[ -z "$INPUT" ]] || INPUT="$(abspath "$INPUT")"
[[ -z "$TRAIN_DATA" ]] || TRAIN_DATA="$(abspath "$TRAIN_DATA")"
if [[ -n "$EVAL_MODEL" && ( -e "$EVAL_MODEL" || "$EVAL_MODEL" == /* || "$EVAL_MODEL" == .* ) ]]; then
  EVAL_MODEL="$(abspath "$EVAL_MODEL")"
fi
EVAL_DATA_ABS=""
if [[ ${#EVAL_DATA[@]} -gt 0 ]]; then
  for f in "${EVAL_DATA[@]}"; do
    f="$(abspath "$f")"; [[ "$f" != *$'\n'* ]] || die "--eval-data path contains a newline"
    EVAL_DATA_ABS="${EVAL_DATA_ABS}${f}"$'\n'
  done
fi

case "$CACHE_ROOT/" in "$HOME"/*) log "warning: cache root $CACHE_ROOT is under \$HOME; prefer scratch storage (--cache-root)" ;; esac

# --------------------------------------------------------------- scheduler helpers
TERMINAL_RE='^(COMPLETED|FAILED|CANCELLED.*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)$'
ACTIVE_RE='^(PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|REQUEUED|REQUEUE_HOLD|REQUEUE_FED|RESIZING|STAGE_OUT|SIGNALING)$'

# job_state ID -> prints STATE|EXIT|REASON; "GONE||" when neither squeue nor sacct know it
# (yet); returns 1 only when the scheduler cannot be queried at all.
job_state() {
  local id="$1" q a i sq_ok=0 sq_err
  sq_err="$(mktemp "${TMPDIR:-/tmp}/vsft-squeue.XXXXXX")"
  if q="$(squeue -h -j "$id" -o '%T|-|%r' 2>"$sq_err")"; then
    sq_ok=1
    q="$(head -n1 <<< "$q")"
    # squeue is authoritative only for active jobs; a finished job lingering in slurmctld
    # (MinJobAge) has no exit code there, so terminal states come from sacct below.
    if [[ -n "$q" && "${q%%|*}" =~ $ACTIVE_RE ]]; then rm -f "$sq_err"; printf '%s' "$q"; return 0; fi
  elif grep -qi 'invalid job id' "$sq_err"; then
    sq_ok=1   # a real answer: the controller no longer knows the job (it left the queue)
  fi
  rm -f "$sq_err"
  # Not active in squeue (or squeue unreachable): ask accounting.
  for i in 1 2 3; do
    if a="$(sacct -j "$id" -X -n -P -o State,ExitCode,Reason 2>/dev/null)"; then
      if [[ -n "$a" ]]; then
        printf '%s' "$(head -n1 <<< "$a" | awk -F'|' '{split($1,s," "); print s[1]"|"$2"|"$3}')"
        return 0
      fi
      # Empty accounting only means GONE if squeue answered too; otherwise unverifiable.
      [[ "$sq_ok" -eq 1 ]] && { printf 'GONE||'; return 0; }
      return 1
    fi
    sleep "$i"
  done
  return 1
}

# reconcile_dependencies IDS -> prints the ids to keep (stdout only); dies otherwise.
reconcile_dependencies() {
  local ids="$1" keep="" id st
  local IFS=':'
  for id in $ids; do
    IFS=' '
    if ! st="$(job_state "$id")"; then
      # A job this script submitted moments ago is trusted while the scheduler is flaky.
      if [[ " ${SUBMITTED_IDS[*]+"${SUBMITTED_IDS[*]}"} " == *" $id "* ]]; then
        log "dependency $id (submitted by this command) not verifiable right now; keeping it"
        keep="${keep:+$keep:}$id"; IFS=':'; continue
      fi
      die "cannot verify dependency $id (scheduler query failed); not submitting"
    fi
    case "${st%%|*}" in
      GONE)
        if [[ " ${SUBMITTED_IDS[*]+"${SUBMITTED_IDS[*]}"} " == *" $id "* ]]; then
          log "dependency $id (submitted by this command) not yet visible; keeping it"
          keep="${keep:+$keep:}$id"
        else
          die "dependency $id unknown to the scheduler (purged or accounting delay); not submitting"
        fi ;;
      COMPLETED)
        [[ "$(cut -d'|' -f2 <<< "$st")" == "0:0" ]] || die "dependency $id completed with exit $(cut -d'|' -f2 <<< "$st")"
        log "dependency $id already COMPLETED|0:0; omitted" ;;
      *)
        if [[ "${st%%|*}" =~ $ACTIVE_RE ]]; then
          keep="${keep:+$keep:}$id"
        else
          die "dependency $id is ${st%%|*}; downstream stages would never run"
        fi ;;
    esac
    IFS=':'
  done
  printf '%s' "$keep"
}

# Reconcile user-supplied dependencies before anything is created on disk.
DEP=""
if [[ -n "$DEPEND_ON" ]]; then
  if [[ "$DRY_RUN" -eq 1 ]]; then DEP="$DEPEND_ON"; else DEP="$(reconcile_dependencies "$DEPEND_ON")"; fi
fi

# ------------------------------------------------------------------- run directory
TS="$(date -u +%Y%m%dT%H%M%SZ)"
check_disk() {
  local dir="$1" avail_kb
  mkdir -p "$dir" 2>/dev/null || true
  avail_kb="$(df -Pk "$dir" | awk 'NR==2 {print $4}')"
  [[ -n "$avail_kb" ]] || die "cannot determine free space at $dir"
  (( avail_kb / 1048576 >= MIN_FREE_GB )) || die "only $((avail_kb / 1048576)) GB free at $dir (< --min-free-gb $MIN_FREE_GB)"
}

snapshot_code() {  # copy what jobs execute, so later edits/syncs never change a queued run
  local dst="$1/code"
  mkdir -p "$dst"
  cp -R "$PROJECT_DIR/src" "$PROJECT_DIR/slurm" "$dst/"
  cp "$PROJECT_DIR/pyproject.toml" "$PROJECT_DIR/uv.lock" "$dst/"
  [[ -f "$PROJECT_DIR/README.md" ]] && cp "$PROJECT_DIR/README.md" "$dst/"
  find "$dst" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  if [[ -n "$ACC_CONFIG" ]]; then cp "$ACC_CONFIG" "$dst/accelerate.yaml"; fi
  chmod -R a-w "$dst/src" "$dst/slurm" 2>/dev/null || true
}

if [[ -n "$RESUME" ]]; then
  RUN_DIR="$(abspath "$RESUME")"
  [[ -f "$RUN_DIR/manifest.json" ]] || die "--resume $RUN_DIR has no manifest.json"
  [[ -f "$RUN_DIR/config.yaml" ]] || die "--resume $RUN_DIR has no config.yaml"
  [[ -f "$RUN_DIR/code/uv.lock" && -d "$RUN_DIR/code/src/vsft" ]] || die "--resume $RUN_DIR has no code snapshot"
  if [[ "$REFRESH_CODE" -eq 1 ]]; then
    # Queued and running jobs read src/ and slurm/lib from the snapshot at run time, so it
    # may only be swapped while nothing of this run is active.
    for f in "$RUN_DIR"/status/*.state; do
      [[ -f "$f" ]] || continue
      jid="$(python3 "$RECEIPT_PY" get "$f" job_id)"
      [[ -n "$jid" && "$jid" != "null" && "$jid" =~ ^[0-9]+$ ]] || continue
      if [[ "$DRY_RUN" -eq 0 ]]; then
        # Unknown is not safe: only swap when the scheduler confirms the job is not active.
        st="$(job_state "$jid")" \
          || die "--refresh-code: cannot verify job $jid ($(basename "${f%.state}")) with the scheduler; not swapping code"
        if [[ "${st%%|*}" =~ $ACTIVE_RE ]]; then
          die "--refresh-code: job $jid ($(basename "${f%.state}")) of this run is ${st%%|*}; wait or scancel it first"
        fi
      fi
    done
    if [[ "$DRY_RUN" -eq 1 ]]; then
      log "dry run: would replace $RUN_DIR/code with a snapshot of $PROJECT_DIR"
    else
      prev="$RUN_DIR/code.prev-$(date -u +%Y%m%dT%H%M%SZ)"
      chmod -R u+w "$RUN_DIR/code" 2>/dev/null || true
      mv "$RUN_DIR/code" "$prev" || die "could not move the old snapshot aside"
      [[ -n "$ACC_CONFIG" || ! -f "$prev/accelerate.yaml" ]] || ACC_CONFIG="$prev/accelerate.yaml"
      snapshot_code "$RUN_DIR"
      python3 "$RECEIPT_PY" append "$RUN_DIR/manifest.json" code_refreshes \
        "$(date -u +%Y%m%dT%H%M%SZ):$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unknown):previous=$prev"
      log "refreshed code snapshot from $PROJECT_DIR (old one kept at $prev)"
    fi
  fi
else
  CONFIG="$(abspath "${CONFIG:-configs/h100/math_qwen35.yaml}")"
  [[ -f "$CONFIG" ]] || die "config not found: $CONFIG"
  if [[ "$STAGE" == "paper" ]]; then
    base="$(abspath "${RUN_BASE:-runs/paper-$(basename "${CONFIG%.*}")}")"   # its own run dir
  else
    base="$(abspath "${RUN_BASE:-runs/$(basename "${CONFIG%.*}")}")"
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    RUN_DIR="$base"
  else
    mkdir -p "$(dirname "$base")" || die "cannot create $(dirname "$base")"
    for i in $(seq 0 999); do
      cand="$base"; [[ "$i" -gt 0 ]] && cand="$(printf '%s-%03d' "$base" "$i")"
      if mkdir "$cand" 2>/dev/null; then RUN_DIR="$cand"; break; fi  # atomic claim
    done
    [[ -n "$RUN_DIR" ]] || die "could not claim a run directory under $base"
  fi
fi
RUN_CONFIG="$RUN_DIR/config.yaml"
CODE_DIR="$RUN_DIR/code"
LOG_DIR="$RUN_DIR/logs"
sha256_of() { python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$1"; }
if [[ -z "$VENV" ]]; then
  lock_src="$CODE_DIR/uv.lock"; [[ -f "$lock_src" ]] || lock_src="$PROJECT_DIR/uv.lock"
  VENV="$CACHE_ROOT/venvs/$(sha256_of "$lock_src" | cut -c1-12)"
fi

if [[ "$DRY_RUN" -eq 0 ]]; then
  mkdir -p "$LOG_DIR" "$RUN_DIR/receipts" "$RUN_DIR/status" "$RUN_DIR/args"
  [[ -w "$LOG_DIR" ]] || die "log dir not writable"
  check_disk "$RUN_DIR"
  check_disk "$CACHE_ROOT"
  if [[ -z "$RESUME" ]]; then
    cp "$CONFIG" "$RUN_CONFIG"
    snapshot_code "$RUN_DIR"
    python3 "$RECEIPT_PY" update "$RUN_DIR/manifest.json" \
      "created_utc=$TS" "project_dir=$PROJECT_DIR" "run_dir=$RUN_DIR" "code_dir=$CODE_DIR" \
      "source_config=$CONFIG" "config=$RUN_CONFIG" \
      "git_commit=$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)" \
      "git_dirty=$([[ -n "$(git -C "$PROJECT_DIR" status --porcelain 2>/dev/null)" ]] && echo true || echo false)"
  fi
  # Inputs that must exist now because nothing earlier in this submission produces them.
  needs_existing() { [[ -z "$DEP" ]] && ! has_stage "$1"; }
  if has_stage boost && needs_existing prepare; then
    f="${INPUT:-$RUN_DIR/data/math_l345_train.jsonl}"; [[ -s "$f" ]] || die "boost input missing: $f (run prepare or pass --input)"
  fi
  if has_stage train && needs_existing boost; then
    f="${TRAIN_DATA:-$RUN_DIR/boosted.jsonl}"; [[ -s "$f" ]] || die "train data missing: $f"
  fi
  if [[ -n "$TRAIN_RESUME" ]]; then
    TRAIN_RESUME="$(abspath "$TRAIN_RESUME")"
    [[ -f "$TRAIN_RESUME/trainer_state.json" ]] || die "--train-resume has no trainer_state.json"
    case "$TRAIN_RESUME" in "$RUN_DIR"/*) ;; *) die "--train-resume must be inside the run dir" ;; esac
  fi
  if has_stage eval && needs_existing train && [[ -z "$EVAL_MODEL" ]]; then
    td="$(cat "$RUN_DIR/status/train.dir" 2>/dev/null || true)"
    [[ -n "$td" && -f "$td/SUCCEEDED.json" ]] || die "no successfully trained model in run (pass --eval-model or include train)"
  fi
  if has_stage paper_sample && needs_existing prepare; then
    f="$RUN_DIR/data/math_l345_train.jsonl"; [[ -s "$f" ]] || die "paper sampler input missing: $f (run prepare)"
  fi
  if has_stage train_paper && needs_existing paper_sample; then
    f="${TRAIN_DATA:-$RUN_DIR/paper_sampled.jsonl}"; [[ -s "$f" ]] || die "train_paper data missing: $f (run paper_sample)"
  fi
  if has_stage eval_paper && needs_existing train_paper && [[ -z "$EVAL_MODEL" ]]; then
    td="$(cat "$RUN_DIR/status/train_paper.dir" 2>/dev/null || true)"
    [[ -n "$td" && -f "$td/SUCCEEDED.json" ]] || die "no successfully trained paper-method model in run (include train_paper)"
  fi
  if has_stage train_sft && needs_existing boost; then
    f="${TRAIN_DATA:-$RUN_DIR/boosted.jsonl}"; [[ -s "$f" ]] || die "plain-SFT data missing: $f (it trains on the boosted rows' expert traces; run boost or pass --train-data)"
  fi
  for s in eval eval_sft eval_base eval_paper; do
    # Test sets must exist now unless prepare is part of this submission.
    if has_stage "$s" && ! has_stage prepare && [[ -z "$DEP" && ${#EVAL_DATA[@]} -eq 0 ]]; then
      for f in math_l345_test math500_test gsm8k_test; do
        [[ -s "$RUN_DIR/data/$f.jsonl" ]] || die "$s: test set missing: $RUN_DIR/data/$f.jsonl (run prepare)"
      done
    fi
  done
  if has_stage eval_sft && needs_existing train_sft && [[ -z "$EVAL_MODEL" ]]; then
    td="$(cat "$RUN_DIR/status/train_sft.dir" 2>/dev/null || true)"
    [[ -n "$td" && -f "$td/SUCCEEDED.json" ]] || die "no successfully trained plain-SFT model in run (include train_sft)"
  fi
  if ! has_stage setup; then
    # Cheap login-node check so a stale/missing venv fails now, not after hours in queue.
    [[ -x "$VENV/bin/python" ]] || die "venv missing at $VENV: submit the setup stage (or slurm/fetch_online.sh) first"
    [[ "$(cat "$VENV/.vsft-uv-lock.sha256" 2>/dev/null)" == "$(sha256_of "$CODE_DIR/uv.lock")" ]] \
      || die "venv $VENV was not built from this run's uv.lock: include setup or pass the matching --venv"
  fi
  # Duplicate guard: never run two copies of a stage on the same run concurrently.
  if [[ "$ALLOW_DUP" -eq 0 ]]; then
    for s in "${STAGES[@]}"; do
      prev="$(python3 "$RECEIPT_PY" get "$RUN_DIR/status/$s.state" job_id)"
      if [[ -n "$prev" && "$prev" != "null" ]]; then
        st="$(job_state "$prev" || true)"
        if [[ "${st%%|*}" =~ $ACTIVE_RE ]]; then
          die "stage $s of this run is already ${st%%|*} as job $prev (use --allow-duplicate to override)"
        fi
      fi
    done
  fi
fi

# A separate verifier shares the boost GPU with the student: that needs an 80 GB H100.
# Checked here (stdlib YAML subset reader) so a smaller GPU never wastes the queue wait.
cfg_for_check="$RUN_CONFIG"; [[ -f "$cfg_for_check" ]] || cfg_for_check="${CONFIG:-}"
if has_stage boost && [[ -n "$cfg_for_check" && -f "$cfg_for_check" ]]; then
  sep_verifier="$(python3 - "$cfg_for_check" <<'PY'
import re, sys
sec, vals = None, {}
for line in open(sys.argv[1]):
    line = line.split("#", 1)[0].rstrip()
    if not line.strip():
        continue
    m = re.match(r"^(\w+):\s*$", line)
    if m:
        sec = m.group(1); continue
    m = re.match(r"^\s+(\w+):\s*(.*)$", line)
    if m and sec:
        vals[(sec, m.group(1))] = m.group(2).strip().strip("'\"")
v = vals.get(("verifier", "model"), "")
s = vals.get(("model", "name_or_path"), "")
print("1" if v and v != s else "0")
PY
)"
  if [[ "$sep_verifier" == "1" ]] && awk -v m="$MIN_GPU_MEM_GB" 'BEGIN{exit !(m < 79)}'; then
    die "this config runs a separate verifier next to the student on the boost GPU, which needs an 80 GB H100: pass --min-gpu-mem-gb 79 (the default) and use an 80 GB H100 partition"
  fi
fi

# ---------------------------------------------------------------- submission
FIRST_JOB="" FIRST_STAGE="" FIRST_RECEIPT="" FIRST_LOG="" LAST_ID=""

submit_stage() {  # submit_stage STAGE DEPENDENCY_IDS
  local stage="$1" dep="$2" gpu=1 receipt token log_out log_err args_file
  case "$stage" in setup|prepare) gpu=0 ;; esac
  token="vsft-$(python3 -c 'import uuid; print(uuid.uuid4().hex)')"
  local tag="$stage-$TS-${token:5:8}"
  receipt="$RUN_DIR/receipts/$tag.json"
  log_out="$LOG_DIR/$stage-$TS-%j.out"
  log_err="$LOG_DIR/$stage-$TS-%j.err"
  args_file="$RUN_DIR/args/$tag.args"

  local sb=(sbatch --parsable --account "$ACCOUNT" --job-name "vsft-$stage" --chdir "$RUN_DIR"
            --output "$log_out" --error "$log_err" --comment "$token" --export ALL)
  if [[ "$gpu" -eq 1 ]]; then
    local gpus="$GPUS"
    case "$stage" in train|train_sft|train_paper) gpus="${TRAIN_GPUS:-$GPUS}" ;; esac
    sb+=(--gpus-per-node "$gpus" --cpus-per-task "$CPUS" --mem "$MEM")
    [[ -n "$PARTITION" ]] && sb+=(--partition "$PARTITION")
    [[ -n "$TIME" ]] && sb+=(--time "$TIME")
    [[ -n "$CONSTRAINT" ]] && sb+=(--constraint "$CONSTRAINT")
  else
    [[ -n "${CPU_PARTITION:-$PARTITION}" ]] && sb+=(--partition "${CPU_PARTITION:-$PARTITION}")
    [[ -n "$CPU_MEM" ]] && sb+=(--mem "$CPU_MEM")
    [[ -n "$CPU_TIME" ]] && sb+=(--time "$CPU_TIME")
  fi
  [[ -n "$QOS" ]] && sb+=(--qos "$QOS")
  [[ -n "$RESERVATION" ]] && sb+=(--reservation "$RESERVATION")
  [[ -n "$dep" ]] && sb+=(--dependency "afterok:$dep" --kill-on-invalid-dep=yes)
  sb+=("$CODE_DIR/slurm/jobs/$(script_of "$stage").sbatch")

  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] %s:' "$stage"; printf ' %q' "${sb[@]}"; printf '\n'
    LAST_ID="DRYRUN_$stage"; SUBMITTED_IDS+=("$LAST_ID"); return 0
  fi

  : > "$args_file"
  if [[ ${#EXTRA[@]} -gt 0 ]]; then printf '%s\0' "${EXTRA[@]}" > "$args_file"; fi

  # Job environment (sbatch --export ALL forwards these; no comma-escaping issues).
  export VSFT_PROJECT_DIR="$PROJECT_DIR" VSFT_CODE_DIR="$CODE_DIR" VSFT_RUN_DIR="$RUN_DIR"
  export VSFT_STAGE="$stage" VSFT_RECEIPT="$receipt" VSFT_ARGS_FILE="$args_file"
  export VSFT_CONFIG="$RUN_CONFIG" VSFT_VENV="$VENV" VSFT_CACHE_ROOT="$CACHE_ROOT"
  export VSFT_HF_HOME="$HF_HOME_DIR" VSFT_MODULES="${MODULES[*]+"${MODULES[*]}"}"
  export VSFT_OFFLINE="$OFFLINE" VSFT_GPU_TYPE="$GPU_TYPE" VSFT_MIN_GPU_MEM_GB="$MIN_GPU_MEM_GB"
  export VSFT_GPU_CAPABILITY="$GPU_CAPABILITY"
  export VSFT_PREEMPT_GRACE="$PREEMPT_GRACE" VSFT_INPUT="$INPUT" VSFT_TRAIN_DATA="$TRAIN_DATA"
  local acc_cfg="$ACC_CONFIG"
  if [[ -n "$acc_cfg" ]]; then
    # Fresh runs use the snapshot copy; resumed runs copy it in now (if not there yet).
    if [[ -f "$CODE_DIR/accelerate.yaml" ]]; then
      cmp -s "$ACC_CONFIG" "$CODE_DIR/accelerate.yaml" \
        || die "--accelerate-config differs from this run's snapshot ($CODE_DIR/accelerate.yaml); start a fresh run"
    else
      cp "$ACC_CONFIG" "$CODE_DIR/accelerate.yaml"
    fi
    acc_cfg="$CODE_DIR/accelerate.yaml"
  fi
  export VSFT_TRAIN_RESUME="$TRAIN_RESUME" VSFT_ACCELERATE_CONFIG="$acc_cfg"
  export VSFT_EVAL_MODEL="$EVAL_MODEL" VSFT_EVAL_DATA="$EVAL_DATA_ABS" VSFT_DATASETS="$DATASETS"

  python3 "$RECEIPT_PY" claim "$receipt" "stage=$stage" "submitted_utc=$TS" \
    "submission_token=$token" "run_dir=$RUN_DIR" "code_dir=$CODE_DIR" "config=$RUN_CONFIG" \
    "command=$(printf '%q ' "${sb[@]}")" "args_file=$args_file" \
    "log_out=$log_out" "log_err=$log_err" "dependency=${dep:-none}" "state=SUBMITTING" \
    "env=$(python3 -c 'import json,os; print(json.dumps({k: v for k, v in sorted(os.environ.items()) if k.startswith("VSFT_")}))')" \
    || die "could not create receipt $receipt"

  local out err rc=0 errf
  errf="$(mktemp "${TMPDIR:-/tmp}/vsft-sbatch.XXXXXX")"
  out="$("${sb[@]}" 2>"$errf")" || rc=$?
  err="$(cat "$errf")"; rm -f "$errf"
  [[ -z "$err" ]] || log "sbatch stderr: $err"
  if [[ "$rc" -ne 0 ]]; then
    python3 "$RECEIPT_PY" update "$receipt" "state=SUBMISSION_REJECTED" "sbatch_exit=$rc" "sbatch_stderr=$err"
    echo "SUBMISSION_REJECTED: sbatch exited $rc for stage $stage: $err" >&2
    echo "receipt: $receipt" >&2
    [[ ${#SUBMITTED_IDS[@]} -eq 0 ]] || echo "already submitted: ${SUBMITTED_IDS[*]}" >&2
    exit 4
  fi
  local jobid
  jobid="$(grep -E '^[0-9]+(;[A-Za-z0-9_.-]+)?$' <<< "$out" | tail -n1 | cut -d';' -f1 || true)"
  if [[ -z "$jobid" ]]; then
    # Ambiguous response: reconcile via the unique comment token, never resubmit blindly.
    local tries
    for tries in 1 2 3; do
      jobid="$(squeue --me -h -o '%i|%k' 2>/dev/null | awk -F'|' -v t="$token" '$2==t {print $1}' | head -n1 || true)"
      [[ -n "$jobid" ]] && break
      sleep "$tries"
    done
    if [[ -z "$jobid" ]]; then
      python3 "$RECEIPT_PY" update "$receipt" "state=UNKNOWN" "sbatch_stdout=$out"
      echo "UNKNOWN/UNVERIFIED: sbatch returned no parsable job id for $stage (stdout: $out)." >&2
      echo "Do NOT resubmit blindly. Check: squeue --me -o '%i %j %k' | grep $token" >&2
      echo "receipt: $receipt" >&2
      exit 3
    fi
    log "reconciled job id $jobid from submission token"
  fi
  python3 "$RECEIPT_PY" update "$receipt" "job_id=$jobid" "state=SUBMITTED" \
    "log_out=${log_out//%j/$jobid}" "log_err=${log_err//%j/$jobid}"
  SLURM_JOB_ID="$jobid" python3 "$RECEIPT_PY" mark "$RUN_DIR" "$stage" SUBMITTED "receipt=$receipt"
  python3 "$RECEIPT_PY" append "$RUN_DIR/manifest.json" jobs "$stage:$jobid:$TS"
  local st
  if st="$(job_state "$jobid")"; then
    log "stage $stage -> job $jobid (${st%%|*})"
  else
    log "stage $stage -> job $jobid (scheduler record not yet visible)"
  fi
  LAST_ID="$jobid"
  SUBMITTED_IDS+=("$jobid")
  if [[ -z "$FIRST_JOB" ]]; then
    FIRST_JOB="$jobid" FIRST_STAGE="$stage" FIRST_RECEIPT="$receipt" FIRST_LOG="${log_out//%j/$jobid}"
  fi
}

log "run dir: $RUN_DIR"
USER_DEP="$DEP"
STAGE_IDS=" "   # " stage:jobid " pairs for the compare graph (bash 3.2: no assoc arrays)
id_of() { local x; for x in $STAGE_IDS; do [[ "${x%%:*}" == "$1" ]] && { echo "${x#*:}"; return; }; done; }
first=1
for s in "${STAGES[@]}"; do
  if [[ "$STAGE" == "compare" ]]; then
    # Depend on this stage's graph predecessor if it was submitted now, else on --depend-on.
    p="$(compare_pred_of "$s")"
    DEP="$USER_DEP"
    if [[ -n "$p" ]] && has_stage "$p"; then DEP="$(id_of "$p")"; fi
    if [[ "$DRY_RUN" -eq 0 && -n "$DEP" ]]; then DEP="$(reconcile_dependencies "$DEP")"; fi
  elif [[ "$first" -eq 0 && "$DRY_RUN" -eq 0 && -n "$DEP" ]]; then
    # Linear chain: reconcile every predecessor immediately before each submission.
    DEP="$(reconcile_dependencies "$DEP")"
  fi
  submit_stage "$s" "$DEP"
  STAGE_IDS="$STAGE_IDS$s:$LAST_ID "
  DEP="$LAST_ID"; first=0
done
[[ "$DRY_RUN" -eq 1 ]] && { log "dry run complete; nothing submitted"; exit 0; }

print_followup() {
  local ids; ids="$(printf '%s,' "${SUBMITTED_IDS[@]}")"; ids="${ids%,}"
  cat <<EOF

RUN_DIR=$RUN_DIR
JOB_IDS=${SUBMITTED_IDS[*]}
RECEIPT=$FIRST_RECEIPT
LOG_PATH=$FIRST_LOG

  slurm/status.sh "$RUN_DIR"
  slurm/logs.sh "$RUN_DIR" $FIRST_STAGE
  slurm/results.sh "$RUN_DIR"
  squeue --job "$ids" --format='%.18i %.14j %.12T %.30R %.10M'
  scontrol show job $FIRST_JOB
  sacct -j $FIRST_JOB --format=JobID,JobName,State,ExitCode,Elapsed,Start,End
  tail -n 200 "$FIRST_LOG"
EOF
}

show_failure() {
  echo "  state|exit|reason: $1" >&2
  echo "  receipt: $FIRST_RECEIPT" >&2
  echo "  log: $FIRST_LOG" >&2
  local f
  for f in "$FIRST_LOG" "${FIRST_LOG%.out}.err"; do
    if [[ -f "$f" ]]; then echo "----- tail $f" >&2; tail -n 40 "$f" >&2; fi
  done
}

# --------------------------------------------------------- bounded startup monitor
if [[ "$NO_MONITOR" -eq 1 ]]; then
  python3 "$RECEIPT_PY" update "$FIRST_RECEIPT" "monitor=SKIPPED"
  log "OUTCOME: UNVERIFIED — monitoring skipped; submitted, startup not observed"
  print_followup; exit 3
fi
marker="$RUN_DIR/status/$FIRST_STAGE.STARTED"
deadline=$(( $(date +%s) + MONITOR_TIMEOUT ))
grace_deadline=$(( deadline + 120 ))   # extra time for sacct to catch up on finished jobs
seen_running=0 last="" failures=0 state=""
while :; do
  if st="$(job_state "$FIRST_JOB")"; then
    failures=0; last="$st"
  else
    failures=$((failures + 1))
  fi
  state="${last%%|*}"
  [[ "$state" == "RUNNING" || "$state" == "COMPLETING" ]] && seen_running=1
  started=0
  if [[ -f "$marker" && "$(python3 "$RECEIPT_PY" get "$marker" job_id)" == "$FIRST_JOB" ]]; then started=1; fi
  stage_state="$(python3 "$RECEIPT_PY" state "$RUN_DIR" "$FIRST_STAGE" "$FIRST_JOB")"

  if [[ "$state" =~ $TERMINAL_RE ]]; then
    if [[ "$state" == "COMPLETED" && "${last#*|}" == 0:0* && "$stage_state" == "SUCCEEDED" ]]; then
      python3 "$RECEIPT_PY" update "$FIRST_RECEIPT" "monitor=COMPLETED"
      log "OUTCOME: STARTED — job $FIRST_JOB already COMPLETED|0:0 and recorded SUCCEEDED"
      print_followup; exit 0
    fi
    python3 "$RECEIPT_PY" update "$FIRST_RECEIPT" "monitor=FAILED" "scheduler_state=$last"
    if [[ "$started" -eq 1 ]]; then
      echo "OUTCOME: FAILED — job $FIRST_JOB ($FIRST_STAGE) started, then ended $state" >&2
    else
      echo "OUTCOME: FAILED — job $FIRST_JOB ($FIRST_STAGE) reached $state before STARTED" >&2
    fi
    show_failure "$last"; exit 1
  fi
  if [[ "$started" -eq 1 && "$seen_running" -eq 1 ]]; then
    python3 "$RECEIPT_PY" update "$FIRST_RECEIPT" "monitor=STARTED"
    log "OUTCOME: STARTED — job $FIRST_JOB ($FIRST_STAGE) initialized on $(python3 "$RECEIPT_PY" get "$marker" host)"
    print_followup; exit 0
  fi
  if [[ "$failures" -ge 5 ]]; then
    python3 "$RECEIPT_PY" update "$FIRST_RECEIPT" "monitor=UNKNOWN" "last_state=${last:-none}"
    echo "OUTCOME: UNKNOWN/UNVERIFIED — scheduler queries keep failing; last state '${last:-none}'." >&2
    echo "  Do NOT resubmit blindly; the job may be queued or running." >&2
    print_followup >&2; exit 3
  fi
  now="$(date +%s)"
  if (( now >= deadline )); then
    reason="$(cut -d'|' -f3 <<< "$last")"
    if [[ "$state" == "PENDING" ]]; then
      python3 "$RECEIPT_PY" update "$FIRST_RECEIPT" "monitor=PENDING" "pending_reason=$reason"
      log "OUTCOME: PENDING — job $FIRST_JOB has not started (reason: ${reason:-?}); it stays queued"
      print_followup; exit 10
    fi
    if [[ "$state" == "GONE" && "$now" -lt "$grace_deadline" ]]; then
      sleep 10; continue   # left the queue but not yet in accounting: wait a little longer
    fi
    python3 "$RECEIPT_PY" update "$FIRST_RECEIPT" "monitor=UNVERIFIED" "last_state=${last:-none}"
    log "OUTCOME: UNKNOWN/UNVERIFIED — job $FIRST_JOB is ${state:-unknown} and has not recorded STARTED yet"
    print_followup; exit 3
  fi
  sleep 10
done
