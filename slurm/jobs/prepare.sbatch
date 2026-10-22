#!/usr/bin/env bash
# CPU job: download datasets into RUN_DIR/data and model weights into HF_HOME (needs internet).
#SBATCH --job-name=vsft-prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
source "${VSFT_CODE_DIR:?submit through slurm/submit.sh}/slurm/lib/job_common.sh"
vsft_init
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
vsft_probe_network "https://huggingface.co/api/models/Qwen/Qwen3.5-0.8B"
vsft_preflight
vsft_mark_started_cpu
DATA_DIR="$VSFT_RUN_DIR/data"
# Datasets come from the run's config (task.prepare) unless --datasets was given.
DATASETS=()
if [[ -n "${VSFT_DATASETS:-}" ]]; then
  read -r -a DATASETS <<< "$VSFT_DATASETS"
else
  while IFS= read -r d; do
    if [[ -n "$d" ]]; then DATASETS+=("$d"); fi
  done < <("$VSFT_PY" -m vsft.hpc config-get --config "$VSFT_CONFIG" task.prepare)
fi
[[ ${#DATASETS[@]} -gt 0 ]] || { _log "no datasets to prepare"; exit 64; }
vsft_run "$VSFT_PY" -m vsft.cli prepare "${DATASETS[@]}" --out-dir "$DATA_DIR"
vsft_run "$VSFT_PY" -m vsft.hpc download-model --config "$VSFT_CONFIG"
for f in "$DATA_DIR"/*.jsonl; do [[ -s "$f" ]] || { _log "empty dataset $f"; exit 3; }; done
vsft_succeed "data_dir=$DATA_DIR"
