#!/usr/bin/env bash
# GPU job: the authors' method (block-wise MCMC sampling, arXiv:2610.02140 Algorithm 1) over
# the same training rows as the other arms. Resumable: rerunning skips finished ids.
#SBATCH --job-name=vsft-paper_sample
#SBATCH --gpus-per-node=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
source "${VSFT_CODE_DIR:?submit through slurm/submit.sh}/slurm/lib/job_common.sh"
vsft_init
TRAIN_FILE="$("$VSFT_PY" -m vsft.hpc config-get --config "$VSFT_CONFIG" task.train_file)"
INPUT="${VSFT_INPUT:-$VSFT_RUN_DIR/data/$TRAIN_FILE}"
OUTPUT="$VSFT_RUN_DIR/paper_sampled.jsonl"
# Student and (if separate) verifier must both be in the offline HF cache.
REQ=(--require-file "$INPUT" --require-file "$VSFT_CONFIG")
MODELS="$("$VSFT_PY" -m vsft.hpc config-get --config "$VSFT_CONFIG" models)"   # errexit applies
[[ -n "$MODELS" ]] || { _log "config names no model"; exit 64; }
while IFS= read -r m; do
  if [[ -n "$m" ]]; then REQ+=(--require-model "$m"); fi
done <<< "$MODELS"
vsft_preflight gpu "${REQ[@]}"
vsft_run "$VSFT_PY" -m vsft.cli paper-sample --config "$VSFT_CONFIG" \
  --input "$INPUT" --output "$OUTPUT" ${VSFT_STAGE_ARGS[@]+"${VSFT_STAGE_ARGS[@]}"}
[[ -s "${OUTPUT%.jsonl}.summary.json" ]] || { _log "boost summary missing"; exit 3; }
vsft_succeed "output=$OUTPUT" "summary=${OUTPUT%.jsonl}.summary.json"
