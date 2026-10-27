# MIST

Codebase for the SHOC2 containerised version of [MIST](https://github.com/BattModels/mist)
(Molecular Insight SMILES Transformer) — a family of molecular foundation
models for property prediction. Files:

- `mist.def`: Apptainer definition file to build a container for MIST
- `build_mist.sh`: Bash script to build the MIST container, pre-download the checkpoints and register the kernel
- `mist.ipynb`: Jupyter notebook to use MIST interactively
- `run_mist.sl`: SLURM script to run `mist.ipynb` non-interactively

Suggested usage:

1. Run `./build_mist.sh` and make sure `mist.sif` is built
2. Go to https://ondemand.nesi.org.nz/public/ and select Jupyter Lab
    - Cluster: SLURM HPC
    - Project Code: uoa04517
    - JupyterLab Module: 2026.7.0-foss-2026-4.6.0
    - Number of Hours: 2
    - Number of Cores: 4
    - Memory per Job: 20 GB
    - GPU: L4
3. When Open OnDemand starts, select the file `mist.ipynb` from the chooser
4. Modify the file to run your workflow
5. If you need to run for longer, or a GPU is not available, save changes in `mist.ipynb`, open a terminal in Open OnDemand, and type:
6. `sbatch run_mist.sl`

## What MIST does

MIST models were pre-trained with masked language modelling on
[Smirk](https://github.com/BattModels/smirk)-tokenized SMILES from Enamine REAL
Space, then fine-tuned for downstream prediction tasks. The notebook
demonstrates molecular property prediction (boiling, flash and melting points)
and scent prediction, both adapted from the upstream
[mist-demo](https://github.com/BattModels/mist-demo) tutorials.

Paper: [arXiv:2510.18900](https://arxiv.org/abs/2510.18900) ·
Models: [huggingface.co/mist-models](https://huggingface.co/mist-models)

## How this differs from upstream

**Upstream already ships an Apptainer definition**, at
[`container/training.def`](https://github.com/BattModels/mist/blob/main/container/training.def).
This one departs from it in three ways:

1. **Plain `ubuntu:22.04` instead of `nvcr.io/nvidia/pytorch`.** The NGC base
   image is roughly 20 GB and exists to support distributed pre-training. The
   SHOC2 use case is inference and fine-tuning from published checkpoints,
   where the torch wheel's own vendored CUDA userspace is enough and
   `apptainer --nv` supplies the host driver.
2. **No SSH agent forwarding.** Upstream's `%post` forwards `$SSH_AUTH_SOCK`
   and runs `ssh-keyscan` against GitHub. Both git dependencies in
   `pyproject.toml` (`deepspeed`, `apetokenizer`) are public `https://` URLs,
   so no credentials are needed and the build stays reproducible in CI.
3. **The project is installed, not just its dependencies.** Upstream uses
   `--no-install-project` because it runs from a mounted working copy. Here
   `electrolyte_fm` is importable from the image so the kernel works without
   any per-user checkout.

`ipykernel`, `ipywidgets` and `matplotlib` are added on top, since MIST's dev
dependency group carries test and lint tooling instead.

A Rust toolchain is installed before `uv sync`, in case no `smirk` wheel
matches the interpreter, and removed afterwards — it is build-time only and
costs well over a gigabyte.

### What the lockfile actually resolves to

Worth knowing, because it does not match the README's compatibility table
upstream: `pyproject.toml` asks for `torch~=2.6`, and `uv.lock` pins
**torch 2.9.0+cu128** on **Python 3.13**, whereas the upstream table lists
PyTorch 2.6. The lock is what gets installed, and `--frozen` means it is
installed verbatim with no re-resolution, so builds are reproducible — just
be aware the running versions are newer than the documented ones. The venv
comes out around 7.5 GB, mostly vendored CUDA 12.8 wheels, so expect a `.sif`
in the 4 GB range.

## Checkpoints are pre-downloaded, not fetched at runtime

NeSI compute nodes have no outbound internet access, so
`AutoModel.from_pretrained()` cannot reach huggingface.co from inside a batch
job. `build_mist.sh` downloads the checkpoints on a login node into
`hf_cache/` in project space, and the container sets `HF_HOME` to point at the
bind-mounted copy with `HF_HUB_OFFLINE=1`.

To add a model, append it to `MODELS` in `build_mist.sh` and re-run that
script from a login node.

Note these checkpoints load with `trust_remote_code=True`, which means the
snapshot includes custom Python modules that execute on load. They come from
the [mist-models](https://huggingface.co/mist-models) organisation; review them
if that matters for your threat model.

## Paths must not contain spaces

Apptainer bind strings are comma-separated and parsed unquoted. A space
anywhere in the project path splits the bind list, Apptainer treats the
fragment after the space as the image path, and the kernel fails with:

```
Kernel died before replying to kernel_info
```

`build_mist.sh` and `run_mist.sl` both check for this and fail with a clear
message rather than a confusing Apptainer error. Keep directory names like
`1_Molecules_generated` rather than `1. Molecules generated`.
