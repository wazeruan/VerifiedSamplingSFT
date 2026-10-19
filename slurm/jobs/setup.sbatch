#!/usr/bin/env bash
# CPU job: build/refresh the shared venv from this run's code snapshot (needs internet).
#SBATCH --job-name=vsft-setup
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
source "${VSFT_CODE_DIR:?submit through slurm/submit.sh}/slurm/lib/job_common.sh"
vsft_init
command -v uv >/dev/null 2>&1 || { _log "uv not on PATH (install it or pass --module)"; exit 66; }
vsft_probe_network "https://pypi.org/simple/"
vsft_mark_started_cpu
mkdir -p "$(dirname "$VSFT_VENV")"
vsft_run uv sync --project "$VSFT_CODE_DIR" --frozen --extra vllm   # into UV_PROJECT_ENVIRONMENT=$VSFT_VENV
vsft_run "$VSFT_PY" -m vsft.hpc stamp-lock --lock "$VSFT_CODE_DIR/uv.lock"
vsft_run "$VSFT_PY" -c "import vsft, torch, transformers, vllm, fla; print('env ok', torch.__version__, transformers.__version__, vllm.__version__, 'fla', fla.__version__)"
vsft_succeed "venv=$VSFT_VENV"
