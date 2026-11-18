# Sourced by the vast.ai scripts. Finds conda and activates an env robustly.
#
# Vast templates differ: some ship their own conda (/opt/miniforge3) with envs in /venv and a plain
# venv ("main") auto-activated from ~/.bashrc. A plain `conda activate` from a second conda install
# can then leave the template's python first on PATH, so we put the env's bin/ first explicitly and
# verify which python is used.

if [ -z "${CONDA_DIR:-}" ]; then
  for c in /opt/miniforge3 "$HOME/miniconda3" /opt/conda; do
    if [ -x "$c/bin/conda" ]; then CONDA_DIR="$c"; break; fi
  done
  CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
fi

load_conda() {
  export PATH="$CONDA_DIR/bin:$PATH"
  # shellcheck disable=SC1091
  source "$CONDA_DIR/etc/profile.d/conda.sh"
}

activate_env() {  # $1 = env name
  local name="$1" prefix
  set +u  # conda activate.d scripts are not nounset-safe
  deactivate 2>/dev/null || true  # template venv, if any
  conda activate "$name"
  set -u
  prefix="$(conda env list | awk -v n="$name" '$1==n {print $NF}')"
  if [ -z "$prefix" ] || [ ! -x "$prefix/bin/python" ]; then
    echo "!! conda env '$name' not found"; return 1
  fi
  export PATH="$prefix/bin:$PATH"
  hash -r
  if [ "$(python -c 'import sys; print(sys.prefix)')" != "$prefix" ]; then
    echo "!! could not activate '$name': python is $(command -v python)"; return 1
  fi
  echo "Using env '$name': $(command -v python)"
}
