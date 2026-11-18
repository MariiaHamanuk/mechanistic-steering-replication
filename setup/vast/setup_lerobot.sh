#!/usr/bin/env bash
# One-time setup for the "Observing and Controlling Features in VLAs" replication (LeRobot, PyTorch,
# pi0.5 + LIBERO) on a vast.ai instance. Safe to re-run.
#   bash setup/vast/setup_lerobot.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
cd "$REPO_ROOT"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
ENV_NAME=lerobot
LEROBOT_VERSION=0.7.0
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"

echo "=== [1/4] System packages ==="
export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -y
$SUDO apt-get install -y --no-install-recommends build-essential git wget tmux ffmpeg \
  libegl1 libegl-dev libgl1 libgl-dev libglib2.0-0

echo "=== [2/4] Miniconda ==="
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
fi
export PATH="$CONDA_DIR/bin:$PATH"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

echo "=== [3/4] Conda env '$ENV_NAME' (python 3.12, lerobot==$LEROBOT_VERSION) ==="
if ! conda env list | grep -qE "^$ENV_NAME\s"; then
  conda create -y -n "$ENV_NAME" python=3.12
fi
set +u; conda activate "$ENV_NAME"; set -u  # conda activate.d scripts are not nounset-safe
export CMAKE_POLICY_VERSION_MINIMUM=3.5
pip install "lerobot[pi,libero]==$LEROBOT_VERSION" scikit-learn matplotlib
# Our code lives in src/: put it on sys.path (no reinstall of our package's deps needed).
echo "$REPO_ROOT/src" > "$(python -c 'import site; print(site.getsitepackages()[0])')/vla_mech_interp_src.pth"

echo "=== [4/4] Sanity check: GPU, lerobot imports, Hugging Face access ==="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "PyTorch does not see the GPU"
print("GPU:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB")
import lerobot, sklearn  # noqa: F401
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
from feature_control.hooks import PrefixTap  # noqa: F401
from huggingface_hub import hf_hub_download
try:
    hf_hub_download("lerobot/pi05_libero", "config.json")
    print("Hugging Face: lerobot/pi05_libero is accessible")
except Exception as e:
    print("!! Cannot download lerobot/pi05_libero:", type(e).__name__, e)
    print("!! If it says gated/401: open https://huggingface.co/lerobot/pi05_libero, accept the license,")
    print("!! then run:  huggingface-cli login   (paste a read token)")
print("lerobot", lerobot.__version__, "OK")
EOF

echo
echo "Setup finished. Next: bash setup/vast/run_feature_control.sh smoke"
