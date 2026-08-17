#!/bin/bash
# Shared configuration for every CARLA pipeline in this repository: sim2real
# collection, closed-loop evaluation and Bench2Drive.
#
# The login profile no longer exports CARLA_ROOT or PYTHONPATH (the 0.9.13 entry
# it used to add is what made `env -u PYTHONPATH` necessary everywhere). This
# file is now the only place they are set, so source it before running anything
# that talks to CARLA:
#
#     source scripts/define_env.sh

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARLA_ROOT="/mnt/disk/tjhu78u/CARLA_0_9_15"
export B2D_ROOT="/mnt/disk/tjhu78u/workspace/test/Bench2Drive"

# One interpreter for everything. pufferlib, the CARLA 0.9.15 client and the
# leaderboard all live in this Python 3.10.16 venv, so the policy no longer has
# to cross into the evaluator as TorchScript -- though b2d/export.py still
# works and the bundle is still what b2d/agent.py loads.
export PICDRIVE_PYTHON="$PROJECT_ROOT/.venv/bin/python"
export B2D_PYTHON="$PICDRIVE_PYTHON"

# `import carla` must resolve to the cp310 wheel installed in the venv. CARLA
# 0.9.15 ships only cp27/cp37 artifacts under PythonAPI/carla/dist, so that
# directory is deliberately NOT on the path. The one thing still needed from the
# CARLA tree is its pure-python `agents` package (demo/closed_loop.py's route planner
# and the leaderboard both import it), which lives directly under PythonAPI/carla.
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$PROJECT_ROOT"

export BUNDLE="$PROJECT_ROOT/artifacts/carla_sim2real/b2d_bundle"

export B2D_ROUTES="dev10"
export B2D_OUTPUT=""
export B2D_GPU="0"
export B2D_PORT="2000"
export B2D_TM_PORT="50000"
export B2D_SUBSET=""
export B2D_VIZ="0"
export B2D_SPECTATOR="chase"
