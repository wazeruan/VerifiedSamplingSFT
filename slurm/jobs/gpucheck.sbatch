#!/usr/bin/env bash
# GPU job: verify the allocation (A100, memory, CUDA, bf16) and the environment only.
#SBATCH --job-name=vsft-gpucheck
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
source "${VSFT_CODE_DIR:?submit through slurm/submit.sh}/slurm/lib/job_common.sh"
vsft_init
vsft_preflight gpu
_mark STARTED "node=$(hostname)"
vsft_run "$VSFT_PY" -c "import vllm, torch; print('vllm', vllm.__version__, 'cuda', torch.version.cuda)"
vsft_succeed
