#!/bin/bash
#SBATCH --job-name=train_nanoVLM
#SBATCH --output=logs/train_nanoVLM/%A_%a.out
#SBATCH --error=logs/train_nanoVLM/%A_%a.err
#SBATCH --time=14:00:00
#SBATCH --gpus=1
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal

cd /fsx/$USER/nanoVLM_encoder_free
source .venv/bin/activate
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")

# -------------- CHANGE ----------
# Old: torchrun --nproc_per_node=1 train.py
#   Bare torchrun defaults to MASTER_PORT=29500. With --gpus=1, SLURM can pack
#   several single-GPU jobs onto one node; they'd all bind 29500 and the second
#   fails with "Address already in use". Derive a unique port per job.
# New:
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 40000 ))
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT train.py
# -------------- END OF CHANGE ----------