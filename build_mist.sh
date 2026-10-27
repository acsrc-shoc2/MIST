#!/bin/bash -e

# Build the MIST container, pre-download the model checkpoints from Hugging
# Face into project space, and register the JupyterLab kernel.

# Location to install the container and work folders into.
# NOTE: no spaces anywhere in this path. Apptainer bind strings are
# comma-separated and parsed unquoted, so a space in a project path silently
# splits the bind list and the kernel fails to start.
PROJDIR=/nesi/project/uoa04517/SHOC2/MIST

# Group/project the data and container should be readable by
GROUP=uoa04517

case "$PROJDIR" in
    *" "*) echo "ERROR: PROJDIR contains a space; Apptainer binds will break." >&2
           exit 1 ;;
esac

mkdir -p "$PROJDIR"
cd "$PROJDIR"

# Set up cache folders
unset APPTAINER_BIND
export APPTAINER_CACHEDIR=$(mktemp -d)
export APPTAINER_TMPDIR=${APPTAINER_CACHEDIR}

# Build container
apptainer build --force mist.sif mist.def
chmod 640 mist.sif
# Should also be available publicly via github actions:
# apptainer pull mist.sif oras://ghcr.io/acsrc-shoc2/mist/mist:latest

APPTAINER_HOME="/opt/mist"

mkdir -p "$PROJDIR/hf_cache" "$PROJDIR/output"

# ---------------------------------------------------------------------------
# Pre-download the checkpoints.
#
# This must happen here, on a login node, rather than on first use in the
# notebook: NeSI compute nodes have no outbound internet access, so an
# AutoModel.from_pretrained() call that needs to reach huggingface.co will
# hang or fail inside a batch job. The container sets HF_HUB_OFFLINE=1 and
# points HF_HOME at this directory, which is bind-mounted back in.
#
# These models use trust_remote_code, so the snapshot includes custom Python
# modules alongside the weights. Review them if that matters to you:
#   https://huggingface.co/mist-models
# ---------------------------------------------------------------------------
MODELS="mist-models/mist-26.9M-b302p09x-bp
mist-models/mist-26.9M-cyuo2xb6-fp
mist-models/mist-26.9M-y3ge5pf9-mp
mist-models/mist-26.9M-48kpooqf-odour
mist-models/mist-28M-ti624ev1"

echo "Pre-downloading MIST checkpoints into $PROJDIR/hf_cache"
for m in $MODELS; do
    echo "  $m"
    # HF_HUB_OFFLINE is baked into the image, so override it for the download.
    apptainer exec \
        --env HF_HUB_OFFLINE=0 \
        --env HF_HOME="$APPTAINER_HOME/hf_cache" \
        -B "$PROJDIR/hf_cache:$APPTAINER_HOME/hf_cache" \
        mist.sif \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('$m')"
done

chgrp -R "$GROUP" "$PROJDIR"
chmod -R g+rwX "$PROJDIR"

# ---------------------------------------------------------------------------
# Register the JupyterLab kernel
# ---------------------------------------------------------------------------
module load JupyterLab/2026.7.0-foss-2026-4.6.0

BIND_STRING="-B $PROJDIR/hf_cache:$APPTAINER_HOME/hf_cache"
BIND_STRING="$BIND_STRING,$PROJDIR/output:$APPTAINER_HOME/output"

rm -rf /nesi/project/uoa04517/.jupyter/share/jupyter/kernels/mist
nesi-add-kernel mist -cp mist.sif \
    --container-args="--nv --pwd $APPTAINER_HOME $BIND_STRING" \
    --shared -a "$GROUP"

echo To bind local folders to inside container, use the following:
echo "$BIND_STRING"
echo "$BIND_STRING" > bind_syntax.txt
