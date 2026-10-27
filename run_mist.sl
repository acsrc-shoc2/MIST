#!/bin/bash -e

#SBATCH --time          1:00:00
#SBATCH --mem           20GB
#SBATCH --cpus-per-task 4
#SBATCH --account       uoa04517
#SBATCH --gpus-per-node A100:1
#SBATCH --job-name      mist
#SBATCH --output        mist.log

# MIST-26.9M is small enough that CPU inference is viable, so for the
# prediction workflows in mist.ipynb you can drop the --gpus-per-node line
# entirely and the notebook will fall back to CPU. A GPU is worth requesting
# for fine-tuning or for large batches of SMILES.

module load JupyterLab/2026.7.0-foss-2026-4.6.0

# Where the shared mist kernel is registered
export JUPYTER_PATH=/nesi/project/uoa04517/.jupyter/share/jupyter

# Apptainer bind strings are comma-separated and parsed unquoted, so a space
# anywhere in the submit directory splits the bind list and the kernel dies
# with "Kernel died before replying to kernel_info". Fail early and clearly.
case "$SLURM_SUBMIT_DIR" in
    *" "*) echo "ERROR: submit directory contains a space: $SLURM_SUBMIT_DIR" >&2
           echo "Rename it (e.g. 1._Foo -> 1_Foo) and resubmit." >&2
           exit 1 ;;
esac

# Compute nodes have no outbound internet access. The checkpoints must already
# be in the bind-mounted Hugging Face cache, put there by build_mist.sh on a
# login node. Fail with a clear message rather than a stalled HTTP request.
HF_CACHE=/nesi/project/uoa04517/SHOC2/MIST/hf_cache
if [ ! -d "$HF_CACHE/hub" ]; then
    echo "ERROR: no Hugging Face cache at $HF_CACHE/hub" >&2
    echo "Run build_mist.sh from a login node first to download the models." >&2
    exit 1
fi

papermill mist.ipynb output_mist.ipynb --kernel mist
