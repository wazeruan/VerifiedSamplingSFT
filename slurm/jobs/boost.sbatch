#!/usr/bin/env bash
# GPU job: verifier-guided boosting of expert traces. Resumable: rerunning skips finished ids;
# on the USR1 time-limit warning it stops between batches (exit 75) without losing rows.
#SBATCH --job-name=vsft-boost
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
source "${VSFT_CODE_DIR:?submit through slurm/submit.sh}/slurm/lib/job_common.sh"
vsft_init
INPUT="${VSFT_INPUT:-$VSFT_RUN_DIR/data/math_l345_train.jsonl}"
OUTPUT="$VSFT_RUN_DIR/boosted.jsonl"
MODEL="$("$VSFT_PY" -m vsft.hpc config-get --config "$VSFT_CONFIG" model.name_or_path)"
vsft_preflight gpu --require-file "$INPUT" --require-file "$VSFT_CONFIG" --require-model "$MODEL"
vsft_run "$VSFT_PY" -m vsft.cli boost --config "$VSFT_CONFIG" \
  --input "$INPUT" --output "$OUTPUT" ${VSFT_STAGE_ARGS[@]+"${VSFT_STAGE_ARGS[@]}"}
[[ -s "${OUTPUT%.jsonl}.summary.json" ]] || { _log "boost summary missing"; exit 3; }
vsft_succeed "output=$OUTPUT" "summary=${OUTPUT%.jsonl}.summary.json"
