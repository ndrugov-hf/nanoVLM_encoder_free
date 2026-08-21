"""Regression guards for eval.slurm SLURM environment handling."""

from pathlib import Path

EVAL_SLURM = Path(__file__).resolve().parents[1] / "eval.slurm"

# Inherited from a parent training job via sbatch --export=ALL. If left set while this
# job's allocation differs (e.g. parent 8 GPU, eval 4 GPU), nested srun --gres=gpu:1
# fails with "Invalid generic resource (gres) specification".
INHERITED_SLURM_GPU_VARS = (
    "SLURM_GPUS",
    "SLURM_JOB_GPUS",
    "SLURM_STEP_GPUS",
    "SLURM_GPUS_PER_NODE",
    "SLURM_GPUS_PER_TASK",
    "SLURM_GRES",
    "SLURM_JOB_GRES",
    "SLURM_STEP_GRES",
)


def test_eval_slurm_unsets_inherited_slurm_gpu_env() -> None:
    """eval.slurm must drop inherited GPU/GRES vars before nested srun steps."""
    text = EVAL_SLURM.read_text(encoding="utf-8")
    unset_section = text.split("unset", 1)[1] if "unset" in text else ""
    for var in INHERITED_SLURM_GPU_VARS:
        assert var in unset_section, (
            f"eval.slurm must unset {var} so nested srun uses this job's allocation"
        )
