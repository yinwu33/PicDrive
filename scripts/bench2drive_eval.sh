#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

export B2D_ROUTES="leaderboard/data/drivetransformer_bench2drive_dev10.xml"
export B2D_OUTPUT="$PROJECT_ROOT/artifacts/carla_sim2real/bench2drive/dev10"
mkdir -p "$B2D_OUTPUT"

export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$B2D_ROOT/leaderboard:$B2D_ROOT/scenario_runner:$PROJECT_ROOT"
export SCENARIO_RUNNER_ROOT="$B2D_ROOT/scenario_runner"
export LEADERBOARD_ROOT="$B2D_ROOT/leaderboard"
export CHALLENGE_TRACK_CODENAME=SENSORS
export IS_BENCH2DRIVE=True
export PLANNER_TYPE=only_traj
export SAVE_PATH="$B2D_OUTPUT/dump"
export PICDRIVE_VIZ="$B2D_VIZ"
export PICDRIVE_SPECTATOR="$B2D_SPECTATOR"
export PICDRIVE_EXTERNAL_CARLA=1
export CUDA_VISIBLE_DEVICES="$B2D_GPU"

cd "$B2D_ROOT"
exec "$B2D_PYTHON" leaderboard/leaderboard/leaderboard_evaluator.py \
  --routes="$B2D_ROUTES" \
  --repetitions=1 \
  --track=SENSORS \
  --agent="$PROJECT_ROOT/data_utils/sim2real/carla/b2d/agent.py" \
  --agent-config="$BUNDLE" \
  --checkpoint="$B2D_OUTPUT/eval.json" \
  --debug-checkpoint="$B2D_OUTPUT/live_results.txt" \
  --resume=True \
  --port="$B2D_PORT" \
  --traffic-manager-port="$B2D_TM_PORT" \
  --gpu-rank="$B2D_GPU"