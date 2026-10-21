#!/usr/bin/env bash
# GPU job: single-shot accuracy on the run's test sets. The stage name picks the model:
#   eval       Sampling SFT model  (status/train.dir)      label "sampling"
#   eval_sft   plain SFT baseline  (status/train_sft.dir)  label "sft"
#   eval_base  untrained base model (config model)          label "base"
#   eval_paper the authors' method model (status/train_paper.dir)  label "paper"
# VSFT_EVAL_MODEL overrides the model for any of them.
#SBATCH --job-name=vsft-eval
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
source "${VSFT_CODE_DIR:?submit through slurm/submit.sh}/slurm/lib/job_common.sh"
vsft_init
case "$VSFT_STAGE" in
  eval)      LABEL="sampling"; TRAIN_STAGE="train" ;;
  eval_sft)  LABEL="sft";      TRAIN_STAGE="train_sft" ;;
  eval_base) LABEL="base";     TRAIN_STAGE="" ;;
  eval_paper) LABEL="paper";   TRAIN_STAGE="train_paper" ;;
  *) _log "eval.sbatch does not handle stage $VSFT_STAGE"; exit 64 ;;
esac
MODEL="${VSFT_EVAL_MODEL:-}"
TRAIN_DIR=""
if [[ -z "$MODEL" && -z "$TRAIN_STAGE" ]]; then
  MODEL="$("$VSFT_PY" -m vsft.hpc config-get --config "$VSFT_CONFIG" model.name_or_path)"
elif [[ -z "$MODEL" ]]; then
  # Gate on the successful model itself (not the latest train job's state, which a later
  # resume/duplicate attempt may have replaced).
  [[ -s "$VSFT_RUN_DIR/status/$TRAIN_STAGE.dir" ]] || { _log "no $TRAIN_STAGE model recorded"; exit 64; }
  TRAIN_DIR="$(cat "$VSFT_RUN_DIR/status/$TRAIN_STAGE.dir")"
  [[ -f "$TRAIN_DIR/SUCCEEDED.json" ]] || { _log "$TRAIN_DIR has no SUCCEEDED.json; refusing to evaluate"; exit 64; }
  MODEL="$TRAIN_DIR/final"
fi
EVAL_DATA=()
if [[ -n "${VSFT_EVAL_DATA:-}" ]]; then
  while IFS= read -r line; do [[ -n "$line" ]] && EVAL_DATA+=("$line"); done <<< "$VSFT_EVAL_DATA"
else
  EVAL_DATA=("$VSFT_RUN_DIR/data/math_l345_test.jsonl" "$VSFT_RUN_DIR/data/math500_test.jsonl"
             "$VSFT_RUN_DIR/data/gsm8k_test.jsonl")
fi
REQ=(--require-model "$MODEL")
for f in "${EVAL_DATA[@]}"; do REQ+=(--require-file "$f"); done
vsft_preflight gpu "${REQ[@]}"
if [[ -n "$TRAIN_DIR" ]]; then "$VSFT_PY" -m vsft.hpc check-train --dir "$TRAIN_DIR"; fi
OUT="$VSFT_RUN_DIR/eval/$LABEL-${SLURM_JOB_ID}"
vsft_run "$VSFT_PY" -m vsft.cli eval --config "$VSFT_CONFIG" --model "$MODEL" --label "$LABEL" \
  --data "${EVAL_DATA[@]}" --output-dir "$OUT" ${VSFT_STAGE_ARGS[@]+"${VSFT_STAGE_ARGS[@]}"}
[[ -s "$OUT/eval_summary.json" ]] || { _log "eval summary missing"; exit 3; }
vsft_succeed "eval_dir=$OUT" "model=$MODEL" "label=$LABEL"
