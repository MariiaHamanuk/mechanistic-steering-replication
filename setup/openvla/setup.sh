#!/usr/bin/env bash
# Writes the LIBERO config (utils/libero_config/config.yaml) and makes sure LIBERO assets are available.
#   - LIBERO installed from GitHub ships its assets inside the package -> those are used, nothing is downloaded.
#   - Otherwise assets are downloaded from HuggingFace into utils/libero_assets/.
#   - The libero_10 *training* dataset is not needed for evaluation; set DOWNLOAD_LIBERO_DATASETS=1 to fetch it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
export REPO_ROOT
export LIBERO_CONFIG_PATH="${REPO_ROOT}/utils/libero_config"
export ASSETS_DIR="${REPO_ROOT}/utils/libero_assets"
export DATASETS_DIR="${REPO_ROOT}/utils/libero_datasets"
export DOWNLOAD_LIBERO_DATASETS="${DOWNLOAD_LIBERO_DATASETS:-0}"

mkdir -p "${LIBERO_CONFIG_PATH}" "${ASSETS_DIR}" "${DATASETS_DIR}"

SCRIPT="$(mktemp --suffix=.py)"
cat > "$SCRIPT" <<'EOF'
import importlib.util
import os
from pathlib import Path

import yaml

assets_dir = Path(os.environ["ASSETS_DIR"])
datasets_dir = Path(os.environ["DATASETS_DIR"])
config_path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"

spec = importlib.util.find_spec("libero.libero")
if spec is None or spec.origin is None:
    raise RuntimeError("libero package not found")
benchmark_root = Path(spec.origin).parent

bundled_assets = benchmark_root / "assets"
if bundled_assets.is_dir() and any(bundled_assets.iterdir()):
    assets_dir = bundled_assets
    print(f"Using LIBERO assets bundled with the package: {assets_dir}")
    need_download = False
else:
    need_download = True

base = {
    "benchmark_root": str(benchmark_root),
    "bddl_files": str(benchmark_root / "bddl_files"),
    "init_states": str(benchmark_root / "init_files"),
    "datasets": str(datasets_dir),
    "assets": str(assets_dir),
}
config_path.parent.mkdir(parents=True, exist_ok=True)
with config_path.open("w", encoding="utf-8") as f:
    yaml.safe_dump(base, f)
print(f"Wrote LIBERO config to {config_path}")

if need_download:
    from libero.libero.utils.download_utils import download_assets_from_huggingface

    download_assets_from_huggingface(download_dir=str(assets_dir))

if os.environ.get("DOWNLOAD_LIBERO_DATASETS") == "1":
    from libero.libero.utils.download_utils import download_from_huggingface

    if not (datasets_dir / "libero_10").exists():
        download_from_huggingface(dataset_name="libero_10", download_dir=str(datasets_dir), check_overwrite=False)
    else:
        print(f"Dataset already present at {datasets_dir / 'libero_10'}")
EOF

if [ "${CONDA_DEFAULT_ENV:-}" = "openvla-interp" ]; then
  python "$SCRIPT"
else
  conda run --no-capture-output -n openvla-interp python "$SCRIPT"
fi
rm -f "$SCRIPT"
