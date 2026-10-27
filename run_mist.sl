#!/bin/bash -le

# -l makes this a login shell so the NeSI arch module (NeSI/zen3) is
# initialised before `module load` runs. Without it the script only works when
# submitted from a session that already has modules loaded, such as an Open
# OnDemand terminal; submitted over plain ssh or from cron it fails with
# "JupyterLab ... cannot be loaded as requested" and papermill: command not found.

#SBATCH --time          1:00:00
#SBATCH --mem           20GB
#SBATCH --cpus-per-task 4
#SBATCH --account       uoa04517
#SBATCH --gpus-per-node l4:1
##SBATCH --gpus-per-node a100:1
#SBATCH --job-name      mist
#SBATCH --output        mist.log

# The container is CUDA 12.8, which the NeSI driver (580.95.05) supports.
#
# l4 is the default because the a100 queue is by far the longest on Mahuika
# and these models do not need it. MIST-26.9M is small enough that CPU
# inference is viable, so you can also drop the --gpus-per-node line entirely
# and the notebook will fall back to CPU. Uncomment the a100 line for
# fine-tuning or large batches of SMILES.

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
