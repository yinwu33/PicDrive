#!/bin/bash

export PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CARLA_ROOT="/mnt/disk/tjhu78u/CARLA_0_9_15"
export B2D_ROOT="$PROJECT_ROOT/third_party/Bench2Drive"

export PICDRIVE_PYTHON="$PROJECT_ROOT/.venv/bin/python"
export B2D_PYTHON="$PICDRIVE_PYTHON"

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
