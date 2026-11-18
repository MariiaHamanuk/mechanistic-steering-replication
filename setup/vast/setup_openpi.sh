#!/usr/bin/env bash
# One-time setup of the π₀-FAST (OpenPI, JAX) environment on a vast.ai instance. Safe to re-run.
#   bash setup/vast/setup_openpi.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/setup/vast/_conda_env.sh"
ENV_NAME=openpi-interp
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "=== [1/4] System packages ==="
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends build-essential git wget tmux

echo "=== [2/4] Miniconda ==="
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
load_conda
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

echo "=== [3/4] Conda env '$ENV_NAME' (JAX + CUDA 12) ==="
export CMAKE_POLICY_VERSION_MINIMUM=3.5
if conda env list | grep -qE "^$ENV_NAME\s"; then
  conda env update -n "$ENV_NAME" -f setup/openpi/environment.openpi.yml
else
  conda env create -f setup/openpi/environment.openpi.yml
fi
activate_env "$ENV_NAME"
# Checkpoints now live on gs://openpi-assets -> gcsfs (matching fsspec). gcsfs drags in a newer
# protobuf that tensorflow 2.16 rejects, so pin protobuf back afterwards.
pip install "gcsfs==$(python -c 'import fsspec; print(fsspec.__version__)')"
# tensorflow-metadata >=1.17 imports google.protobuf.runtime_version (protobuf >=5), which TF 2.16 cannot use.
pip install protobuf==4.25.8 tensorflow-metadata==1.16.1
pip install -e .
# Modern pip's editable finder can miss packages under src/: put src/ on sys.path explicitly.
echo "$REPO_ROOT/src" > "$(python -c 'import site; print(site.getsitepackages()[0])')/vla_mech_interp_src.pth"

echo "=== [4/4] Sanity check: JAX sees the GPU, openpi imports ==="
python - <<'EOF'
import jax
devs = jax.devices()
print("JAX devices:", devs)
assert any(d.platform == "gpu" for d in devs), "JAX does not see the GPU"
import openpi.models.pi0_fast_hooked  # noqa: F401
import tensorflow_datasets  # noqa: F401
from libero_experiments import experiment_log  # noqa: F401
print("openpi OK")
EOF

echo
echo "Setup finished. Next: bash setup/vast/run_pi0.sh smoke"
