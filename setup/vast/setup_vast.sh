#!/usr/bin/env bash
# One-time setup of a fresh vast.ai instance (Ubuntu image with CUDA 12.x) for the OpenVLA / LIBERO experiments.
# Safe to re-run: finished steps are skipped.
#   bash setup/vast/setup_vast.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME=openvla-interp
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "=== [1/5] System packages (headless rendering for MuJoCo) ==="
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends \
  libegl1 libgl1 libosmesa6 libglew-dev libglib2.0-0 ffmpeg git wget tmux

echo "=== [2/5] Miniconda ==="
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
export PATH="$CONDA_DIR/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

echo "=== [3/5] Conda env '$ENV_NAME' (without flash-attn: compiling it can take >1h; the code falls back to sdpa) ==="
if ! conda env list | grep -qE "^$ENV_NAME\s"; then
  sed '/flash-attn/d' setup/openvla/environment.openvla.yml > /tmp/environment.openvla.noflash.yml
  conda env create -f /tmp/environment.openvla.noflash.yml
fi
conda activate "$ENV_NAME"
pip install -e .
if ! python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('libero.libero') else 1)"; then
  echo "pip 'libero' is not the LIBERO benchmark -> installing LIBERO from GitHub"
  pip install "git+https://github.com/Lifelong-Robot-Learning/LIBERO.git"
fi

echo "=== [4/5] LIBERO config + assets ==="
bash setup/openvla/setup.sh

echo "=== [5/5] Sanity check: GPU + one LIBERO scene rendered offscreen ==="
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH="$REPO_ROOT/utils/libero_config"
sanity() {
python - <<'EOF'
import torch
print("CUDA available:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
from libero.libero import benchmark
from libero_experiments.libero_utils import get_libero_env
suite = benchmark.get_benchmark_dict()["libero_10"]()
env, desc = get_libero_env(suite.get_task(0), resolution=256)
env.reset()
obs = env.set_init_state(suite.get_task_init_states(0)[0])
print("LIBERO render OK:", obs["agentview_image"].shape, "|", desc)
EOF
}
if ! sanity; then
  echo "Sanity check failed -> retrying with mujoco==2.3.7 (known-good with robosuite 1.4)"
  pip install "mujoco==2.3.7"
  sanity
fi

echo
echo "Setup finished. Next: bash setup/vast/run_experiments.sh smoke"
