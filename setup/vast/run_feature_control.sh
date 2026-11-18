#!/usr/bin/env bash
# Replication of "Observing and Controlling Features in VLA Models" (Buurmeijer et al. 2026), part 1:
# observability probes + offline steering checks on pi0.5 (LIBERO).
#
#   bash setup/vast/run_feature_control.sh smoke     # tiny run of all 3 steps (~10-15 min incl. 14 GB model download)
#   bash setup/vast/run_feature_control.sh collect   # activations for N_EPISODES episodes
#   bash setup/vast/run_feature_control.sh probe     # linear observers per layer (CPU)
#   bash setup/vast/run_feature_control.sh sweep     # output-sensitivity sweep + closed-form controller check
#   bash setup/vast/run_feature_control.sh all       # collect -> probe -> sweep -> pack
#   bash setup/vast/run_feature_control.sh pack      # ~/results_fc_<date>.tgz (without the big activation files)
#
# Settings (environment variables, optional):
#   POLICY=lerobot/pi05_libero  DATASET=lerobot/libero  N_EPISODES=400  STRIDE=10  RUN=pi05_libero
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../.. && pwd)"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
# shellcheck disable=SC1091
source "$CONDA_DIR/etc/profile.d/conda.sh"
set +u; conda activate lerobot; set -u  # conda activate.d scripts are not nounset-safe
cd "$REPO_ROOT"
export MUJOCO_GL=egl

POLICY="${POLICY:-lerobot/pi05_libero}"
DATASET="${DATASET:-lerobot/libero}"
N_EPISODES="${N_EPISODES:-400}"
STRIDE="${STRIDE:-10}"
RUN="${RUN:-pi05_libero}"
LOGS="feature_control/logs"
S="feature_control/scripts"

smoke() {
  local d="$LOGS/smoke"
  echo ">>> SMOKE: 6 episodes, 48 frames, 2 layers in the sweep"
  python $S/collect.py --policy "$POLICY" --dataset "$DATASET" --n_episodes 6 --stride 10 --max_frames 48 --out "$d" \
    && python $S/probe.py --run "$d" --test_frac 0.34 \
    && python $S/sweep.py --run "$d" --n_frames 8 --layers 0,9 --alphas -0.2,0.2 \
    && echo ">>> SMOKE OK (numbers are meaningless at this size, it only checks the pipeline)"
}
collect() { python $S/collect.py --policy "$POLICY" --dataset "$DATASET" --n_episodes "$N_EPISODES" --stride "$STRIDE" --out "$LOGS/$RUN"; }
probe() { python $S/probe.py --run "$LOGS/$RUN"; }
sweep() { python $S/sweep.py --run "$LOGS/$RUN"; }
pack() {
  local out="$HOME/results_fc_$(date +%Y%m%d_%H%M).tgz"
  tar czf "$out" -C "$REPO_ROOT/feature_control" --exclude='acts_L*.npy' logs
  echo "Packed: $out ($(du -h "$out" | cut -f1))"
}

case "${1:-}" in
  smoke) smoke ;;
  collect) collect ;;
  probe) probe ;;
  sweep) sweep ;;
  all) collect && probe && sweep && pack ;;
  pack) pack ;;
  *) sed -n '2,14p' "$0"; exit 1 ;;
esac
