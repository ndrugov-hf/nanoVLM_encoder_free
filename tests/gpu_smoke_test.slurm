#!/bin/bash
#SBATCH --job-name=vlm-smoke
#SBATCH --output=logs/smoke/%j.out
#SBATCH --error=logs/smoke/%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal

# GPU smoke test for the HF-decoder redesign. Runs the real load_backbone=True paths
# (autocast training step, generate, save/resume) that the CPU unit tests can't cover.
#
#   sbatch tests/gpu_smoke_test.slurm            # both backbones
#   sbatch tests/gpu_smoke_test.slurm --model lfm
#
# Any failed check makes the Python process exit non-zero, so the job ends FAILED.

set -euo pipefail

# A GPU job can inherit distributed env vars from a parent context; clear them so the
# single-process smoke test doesn't try to join a rendezvous.
unset RANK LOCAL_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT NCCL_SOCKET_IFNAME || true

cd /fsx/$USER/nanoVLM_encoder_free
mkdir -p logs/smoke

# HF auth (LFM2.5 may be gated / require an accepted license). Uses the cached token.
if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi
export TOKENIZERS_PARALLELISM=false

# Use the repo's own venv explicitly (avoids the sibling-repo torchrun/venv trap).
echo "Python: $(.venv/bin/python -c 'import sys,torch; print(sys.executable, torch.__version__)')"

# Pass through any sbatch args (e.g. --model lfm); default to both backbones when none given.
if [ "$#" -eq 0 ]; then
  set -- --model both
fi
.venv/bin/python tests/gpu_smoke_test.py "$@"
