#!/usr/bin/env bash
# Run OpenVLA / LIBERO experiments with sheet-compatible logging.
#
#   bash setup/vast/run_experiments.sh smoke      # 1 task, no video: ~3-5 min, checks everything works
#   bash setup/vast/run_experiments.sh baseline   # no intervention (control)
#   bash setup/vast/run_experiments.sh steer      # intervention DICT with coefficient COEF
#   bash setup/vast/run_experiments.sh all        # smoke -> baseline -> steer
#   bash setup/vast/run_experiments.sh pack       # pack all logs into ~/results_<date>.tgz for download
#
# Settings (environment variables, all optional):
#   EXP_AUTHOR="Mariia"  EXP_PRICE_PER_HOUR=0.40   -> written to the sheet row; price enables live $ estimate
#   DICT=up_10  COEF=4.0  TRIALS=1                 -> intervention dict, alpha, trials per task (10 tasks each)
#   EXP_GOAL="..."                                 -> hypothesis text for the sheet
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate openvla-interp

cd "$REPO_ROOT/openvla"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$REPO_ROOT/utils/libero_config"
export TOKENIZERS_PARALLELISM=false

CFG=libero_experiments/configs/runs/example_run.yaml
DICT="${DICT:-up_10}"
COEF="${COEF:-4.0}"
TRIALS="${TRIALS:-1}"
LOGS="libero_experiments/logs"

run_eval() {
  python libero_experiments/scripts/run_eval.py --config "$CFG" "$@"
}

smoke() {
  echo ">>> SMOKE TEST (1 task, no video)"
  run_eval --override intervention.enabled=false --override env.max_tasks=1 \
    --override logging.save_video=false --goal "${EXP_GOAL:-Smoke test: перевірка, що все запускається}"
}

baseline() {
  echo ">>> BASELINE (no intervention), $TRIALS trial(s) per task"
  run_eval --override intervention.enabled=false --override env.num_trials_per_task="$TRIALS" \
    --goal "${EXP_GOAL:-Baseline: success rate OpenVLA на libero_10 без втручання}"
}

steer() {
  echo ">>> STEERING dict=$DICT coef=$COEF, $TRIALS trial(s) per task"
  run_eval --override intervention.enabled=true --override intervention.dict_name="$DICT" \
    --override intervention.coef="$COEF" --override env.num_trials_per_task="$TRIALS" \
    --goal "${EXP_GOAL:-Чи змінює кластер $DICT (α=$COEF) поведінку робота у відповідному напрямку}"
}

pack() {
  local out="$HOME/results_$(date +%Y%m%d_%H%M).tgz"
  (cd "$REPO_ROOT" && git rev-parse HEAD > openvla/$LOGS/commit.txt 2>/dev/null || true)
  tar czf "$out" -C "$REPO_ROOT/openvla" "$LOGS"
  echo "Packed: $out ($(du -h "$out" | cut -f1))"
}

case "${1:-}" in
  smoke) smoke ;;
  baseline) baseline ;;
  steer) steer ;;
  all) smoke && baseline && steer && pack ;;
  pack) pack ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
