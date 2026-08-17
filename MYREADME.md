# Training and CARLA evaluation

Run all client commands from the PicDrive repository root with the local virtual
environment. Start the CARLA server in a separate terminal; do not run the
repository's Python modules from inside the CARLA installation directory.

Everything below assumes one Python **3.10.16** venv and CARLA **0.9.15**, and
every command uses `$PICDRIVE_PYTHON`. Source the environment once per shell —
it is the only place `CARLA_ROOT` and `PYTHONPATH` are set:

```bash
cd /mnt/disk/tjhu78u/workspace/test/PicDrive
source scripts/define_env.sh
```

## Step 1: train the self-play teacher

The local stable setup uses 200-step episodes and removes agents after they
reach their goal instead of respawning them (`goal_behavior = 3`).

```bash
"$PROJECT_ROOT"/.venv/bin/puffer train puffer_giga_3cam \
  --env.episode-length 200 \
  --env.goal-behavior 3
```

Checkpoints are normally written under a timestamped directory:

```text
experiments/puffer_giga_3cam_<timestamp>_<run-id>/model_puffer_giga_3cam_<epoch>.pt
```

Set `CKPT` to the selected teacher checkpoint for every later stage. This one
already exists in the current workspace:

```bash
export CKPT=experiments/skynet/model_puffer_giga_3cam_001400.pt
test -f "$CKPT"
```

## Step 2: train the real-image perception module

Distillation uses three losses:

1. Feature MSE between simulation and real-image perception embeddings.
2. Cosine embedding loss.
3. Planner KL loss between action distributions from the frozen planner.

Each dataset root must have this structure. Keep `processed`, `ego_state`, and
`teacher_features` under the same root:

```text
<data-root>/
  training/{processed,ego_state,teacher_features}/
  validation/{processed,ego_state,teacher_features}/
```

The trainer accepts one root at a time. The CARLA and Waymo recipes below train
separate students; they do not mix the two datasets automatically.

### Option A: CARLA data

#### Load the CARLA environment

Everything runs on CARLA **0.9.15** in a single Python **3.10.16** venv: the
0.9.15 client wheel, `pufferlib`, and the Bench2Drive leaderboard all import in
the same interpreter. `scripts/define_env.sh` is the only place `CARLA_ROOT` and
`PYTHONPATH` are set — the login profile no longer exports them, so the stale
0.9.13 egg that used to force `env -u PYTHONPATH` everywhere is gone.

```bash
source scripts/define_env.sh
"$PICDRIVE_PYTHON" -c "import carla, pufferlib; print(carla.__file__)"
```

The client wheel is already installed. To rebuild the venv from scratch, see
`data_utils/sim2real/carla/README.md`.

#### Start the CARLA server in terminal A

```bash
source scripts/define_env.sh
"$CARLA_ROOT"/CarlaUE4.sh \
  -RenderOffScreen \
  -carla-rpc-port=2000
```

#### Collect training and validation pairs in terminal B

Run from the PicDrive repository root:

```bash
export DATA_ROOT=artifacts/carla_sim2real/sample50k

"$PICDRIVE_PYTHON" -m data_utils.sim2real.carla.collect \
  --output "$DATA_ROOT/training" \
  --town Town01 --town Town02 --town Town10HD \
  --episodes 300 --vehicles 60 --walkers 30 --seed 0 --resume

"$PICDRIVE_PYTHON" -m data_utils.sim2real.carla.collect \
  --output "$DATA_ROOT/validation" \
  --town Town01 --town Town02 --town Town10HD \
  --episodes 50 --vehicles 60 --walkers 30 --seed 0 --resume
```

`collect` creates both `processed/` and `ego_state/`.

#### Extract teacher features and verify the CARLA dataset

Teacher extraction runs abstract simulation images through the frozen teacher
perception encoder. Planner KL is computed later during distillation.

```bash
export DATA_ROOT=artifacts/carla_sim2real/sample50k
export CKPT=experiments/skynet/puffer_giga_3cam_20260813_105022_0ejcqldx.pt

for split in training validation; do
  "$PICDRIVE_PYTHON" \
    -m data_utils.sim2real.waymo.extract_teacher_features \
    --processed "$DATA_ROOT/$split/processed" \
    --checkpoint "$CKPT" \
    --output "$DATA_ROOT/$split/teacher_features" \
    --batch-size 128 \
    --loader-workers 8 \
    --resume 
done

"$PICDRIVE_PYTHON" \
  -m data_utils.sim2real.waymo.verify \
  --root "$DATA_ROOT" \
  --workers 16 \
  --skip-png
```

#### Train the CARLA perception student

The local DINOv2 weights and pinned revision avoid an unplanned download or a
backbone-version change.

```bash
export STUDENT_RUN=artifacts/carla_sim2real/runs/puffer_giga_3cam_distillation

"$PICDRIVE_PYTHON" \
  -m data_utils.sim2real.waymo.train_distillation \
  --root "$DATA_ROOT" \
  --checkpoint "$CKPT" \
  --output "$STUDENT_RUN" \
  --backbone-weights artifacts/carla_sim2real/weights/dinov2_vits14_reg4/model.safetensors \
  --backbone-revision c04b5193082a8d5b0c4856c7937384a48136c5de \
  --epochs 30 \
  --batch-size 4 \
  --accumulation-steps 8 \
  --workers 8 \
  --amp bf16 \
  --wandb-mode online \
  --wandb-name puffer_giga_3cam_distillation
```

Use `--wandb-mode disabled` when W&B is not configured. To continue an
interrupted run, add `--resume "$STUDENT_RUN/last.pt"`; do not start a fresh
run in a non-empty output directory.

### Option B: Waymo data

This is an alternative data source. Only preprocessing and ego-state
extraction read raw TFRecords.

```bash
export DATA_ROOT=artifacts/waymo_sim2real/full
export WAYMO_ROOT=/mnt/disk/data/public/waymo/perception_1_4_3
export CKPT=experiments/skynet/model_puffer_giga_3cam_001400.pt

for split in training validation; do
  .venv/bin/python -m data_utils.sim2real.waymo.preprocess \
    --input "$WAYMO_ROOT/$split" \
    --output "$DATA_ROOT/$split/processed" \
    --workers 8 \
    --resume

  .venv/bin/python -m data_utils.sim2real.waymo.extract_ego_state \
    --input "$WAYMO_ROOT/$split" \
    --output "$DATA_ROOT/$split/ego_state" \
    --workers 8 \
    --resume

  .venv/bin/python -m data_utils.sim2real.waymo.extract_teacher_features \
    --processed "$DATA_ROOT/$split/processed" \
    --checkpoint "$CKPT" \
    --output "$DATA_ROOT/$split/teacher_features" \
    --batch-size 128 \
    --loader-workers 8 \
    --resume
done

.venv/bin/python -m data_utils.sim2real.waymo.verify \
  --root "$DATA_ROOT" \
  --workers 16 \
  --skip-png
```

Train the Waymo student against the same root:

```bash
export DATA_ROOT=artifacts/waymo_sim2real/full
export CKPT=experiments/skynet/model_puffer_giga_3cam_001400.pt
export STUDENT_RUN=artifacts/waymo_sim2real/runs/skynet_001400_dinov2

.venv/bin/python -m data_utils.sim2real.waymo.train_distillation \
  --root "$DATA_ROOT" \
  --checkpoint "$CKPT" \
  --output "$STUDENT_RUN" \
  --backbone-weights artifacts/carla_sim2real/weights/dinov2_vits14_reg4/model.safetensors \
  --backbone-revision c04b5193082a8d5b0c4856c7937384a48136c5de \
  --epochs 30 \
  --batch-size 4 \
  --accumulation-steps 8 \
  --workers 8 \
  --amp bf16 \
  --wandb-mode online \
  --wandb-name skynet_001400_dinov2_waymo
```

# Testing

## CARLA closed-loop demo

Start the CARLA server in terminal A with the command above. Then evaluate from
the repository root in terminal B. This example uses the CARLA student produced
by Option A:

```bash
export CKPT=experiments/skynet/model_puffer_giga_3cam_001400.pt
export STUDENT_RUN=artifacts/carla_sim2real/runs/puffer_giga_3cam_distillation
export EVAL_DIR=artifacts/carla_sim2real/eval/skynet_001400_dinov2_town01_student

source scripts/define_env.sh
"$PICDRIVE_PYTHON" \
  -m data_utils.sim2real.carla.demo.closed_loop \
  --student "$STUDENT_RUN/deployment.pt" \
  --checkpoint "$CKPT" \
  --output "$EVAL_DIR" \
  --town Town01 \
  --episodes 10 \
  --control student \
  --no-shadow \
  --vehicles 40 \
  --walkers 20 \
  --device cuda \
  --amp bf16 \
  --spectator --camera-preview
```

If the output directory already exists, it is overwritten by default. Use
`--resume` only to continue the same evaluation without deleting its existing
results. On a machine with a graphical display, add `--spectator` and
`--camera-preview` for the interactive 2x3 CARLA/Puffer visualization; omit
them for headless runs.
