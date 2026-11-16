#!/usr/bin/env bash
# Run π₀-FAST steering experiments (offline, DROID frames) with sheet-compatible logging.
#
#   bash setup/vast/run_pi0.sh smoke   # fake observation, 1 sample: checks model download + JAX + logging (~5-10 min)
#   bash setup/vast/run_pi0.sh high    # high_cluster on SAMPLES DROID frames
#   bash setup/vast/run_pi0.sh low     # low_cluster  on SAMPLES DROID frames
#   bash setup/vast/run_pi0.sh all     # smoke -> high -> low -> pack
#   bash setup/vast/run_pi0.sh pack    # pack logs into ~/results_pi0_<date>.tgz
#
# Settings (environment variables, all optional):
#   EXP_AUTHOR, EXP_PRICE_PER_HOUR, EXP_GOAL   -> sheet metadata
#   COEF=10  SAMPLES=20  EVERY=10  EPISODE=0   -> alpha, frames per run, frame stride, DROID episode
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
set +u; conda activate openpi-interp; set -u  # conda activate.d scripts are not nounset-safe

cd "$REPO_ROOT"
unset JAX_PLATFORM_NAME  # use the GPU
export TF_CPP_MIN_LOG_LEVEL=2  # tensorflow is only used to read DROID; silence its GPU warnings

CFG=openpi/experiments/configs/runs/example_run.yaml
COEF="${COEF:-10.0}"
SAMPLES="${SAMPLES:-20}"
EVERY="${EVERY:-10}"
EPISODE="${EPISODE:-0}"
LOGS="openpi/experiments/logs"

run_eval() {
  python openpi/experiments/scripts/run_eval.py --config "$CFG" --override logging.run_name=null "$@"
}

smoke() {
  echo ">>> SMOKE TEST (fake observation, 1 sample)"
  run_eval --override env.dataset_name=fake --override intervention.dict_name=low_cluster \
    --override intervention.coef="$COEF" --goal "${EXP_GOAL:-Smoke test π₀-FAST: модель, JAX і логування працюють}"
}

cluster() {
  local name="$1"
  echo ">>> $name, coef=$COEF, $SAMPLES frames every $EVERY steps of DROID episode $EPISODE"
  run_eval --override intervention.enabled=true --override intervention.dict_name="$name" \
    --override intervention.coef="$COEF" --override env.episode_index="$EPISODE" \
    --override env.max_steps=null --override env.show_every="$EVERY" --override env.num_samples="$SAMPLES" \
    --goal "${EXP_GOAL:-Як кластер $name (α=$COEF) змінює дії π₀-FAST на тих самих кадрах DROID}"
}

pack() {
  local out="$HOME/results_pi0_$(date +%Y%m%d_%H%M).tgz"
  tar czf "$out" -C "$REPO_ROOT/openpi/experiments" logs
  echo "Packed: $out ($(du -h "$out" | cut -f1))"
}

case "${1:-}" in
  smoke) smoke ;;
  high) cluster high_cluster ;;
  low) cluster low_cluster ;;
  all) smoke && cluster high_cluster && cluster low_cluster && pack ;;
  pack) pack ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
