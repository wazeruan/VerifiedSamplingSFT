#!/usr/bin/env bash
# GPU job: SFT on the boosted traces. More than one GPU -> accelerate launch (DDP, or FSDP via
# --accelerate-config). On the USR1 warning it saves a resumable state and exits 75.
#SBATCH --job-name=vsft-train
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
source "${VSFT_CODE_DIR:?submit through slurm/submit.sh}/slurm/lib/job_common.sh"
vsft_init
DATA="${VSFT_TRAIN_DATA:-$VSFT_RUN_DIR/boosted.jsonl}"
MODEL="$("$VSFT_PY" -m vsft.hpc config-get --config "$VSFT_CONFIG" model.name_or_path)"
vsft_preflight gpu --require-file "$DATA" --require-file "$VSFT_CONFIG" --require-model "$MODEL"

TRAIN_ARGS=(--config "$VSFT_CONFIG" --data "$DATA")
if [[ -n "${VSFT_TRAIN_RESUME:-}" ]]; then
  # Explicit resume: continue in the directory that owns the state checkpoint.
  [[ -f "$VSFT_TRAIN_RESUME/trainer_state.json" ]] || { _log "bad resume state"; exit 64; }
  VSFT_TRAIN_DIR="$(dirname "$VSFT_TRAIN_RESUME")"
  TRAIN_ARGS+=(--output-dir "$VSFT_TRAIN_DIR" --resume-from "$VSFT_TRAIN_RESUME")
else
  # Fresh run: atomically claim sft, sft-001, ... (never reuse a failed run's directory).
  VSFT_TRAIN_DIR=""
  for i in $(seq 0 999); do
    cand="$VSFT_RUN_DIR/sft"; [[ "$i" -gt 0 ]] && cand="$(printf '%s-%03d' "$cand" "$i")"
    if mkdir "$cand" 2>/dev/null; then VSFT_TRAIN_DIR="$cand"; break; fi
  done
  [[ -n "$VSFT_TRAIN_DIR" ]] || { _log "could not claim a training dir"; exit 73; }
  TRAIN_ARGS+=(--output-dir "$VSFT_TRAIN_DIR")
fi
export VSFT_TRAIN_DIR
python3 "$RECEIPT_PY" update "$VSFT_RECEIPT" "train_dir=$VSFT_TRAIN_DIR"

NGPU="$("$VSFT_PY" -c 'import torch; print(torch.cuda.device_count())')"
if [[ "$NGPU" -gt 1 ]]; then
  LAUNCH=("$VSFT_PY" -m accelerate.commands.launch --num_processes "$NGPU" --num_machines 1)
  if [[ -n "${VSFT_ACCELERATE_CONFIG:-}" ]]; then LAUNCH+=(--config_file "$VSFT_ACCELERATE_CONFIG"); fi
  LAUNCH+=(-m vsft.cli)
else
  LAUNCH=("$VSFT_PY" -m vsft.cli)
fi
vsft_run "${LAUNCH[@]}" train "${TRAIN_ARGS[@]}" ${VSFT_STAGE_ARGS[@]+"${VSFT_STAGE_ARGS[@]}"}
VSFT_PHASE="verify"
"$VSFT_PY" -m vsft.hpc check-train --dir "$VSFT_TRAIN_DIR"
python3 "$RECEIPT_PY" claim "$VSFT_TRAIN_DIR/SUCCEEDED.json" "job_id=$SLURM_JOB_ID" \
  "final_model=$VSFT_TRAIN_DIR/final" "run_dir=$VSFT_RUN_DIR" 2>/dev/null \
  || python3 "$RECEIPT_PY" update "$VSFT_TRAIN_DIR/SUCCEEDED.json" "job_id=$SLURM_JOB_ID" "final_model=$VSFT_TRAIN_DIR/final"
# Record the trained model only now, so a later failed attempt never hides a good one.
printf '%s\n' "$VSFT_TRAIN_DIR" > "$VSFT_RUN_DIR/status/train.dir.tmp"
mv "$VSFT_RUN_DIR/status/train.dir.tmp" "$VSFT_RUN_DIR/status/train.dir"
vsft_succeed "train_dir=$VSFT_TRAIN_DIR" "final_model=$VSFT_TRAIN_DIR/final" "num_gpus=$NGPU"
