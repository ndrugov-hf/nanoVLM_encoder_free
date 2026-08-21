#!/bin/bash
#SBATCH --job-name=train_nanoVLM_torchrun
#SBATCH --output=logs/train_nanoVLM/%A.out
#SBATCH --error=logs/train_nanoVLM/%A.err
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal

# Change to project directory
cd /fsx/$USER/nanoVLM_encoder_free

# Activate virtual environment
source .venv/bin/activate
export HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")

# Run using torchrun
# -------------- CHANGE ----------
# Old: torchrun --nproc_per_node=8 train.py
#   Bare torchrun defaults to MASTER_PORT=29500. Give each job a unique port
#   derived from SLURM_JOB_ID so parallel runs never collide on the same node.
# New:
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 40000 ))
torchrun --nproc_per_node=8 --master_port=$MASTER_PORT train.py
# -------------- END OF CHANGE ----------