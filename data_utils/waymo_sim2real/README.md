# Waymo sim-to-real paired-data pipeline

The pipeline has a strict raw-data boundary:

1. `preprocess.py` and `extract_ego_state.py` are the only stages that read
   Waymo Perception TFRecords.
2. `extract_teacher_features.py` reads compact processed NPZ samples, renders
   their abstract scene, and extracts the frozen 256-D `DriveCam` scene feature.
3. `visualize.py` reads the two derived datasets and makes a 3x3 audit PNG.
4. `train_distillation.py` reads only derived artifacts and the pinned teacher
   checkpoint; it never renders and never updates the simulation policy.

## 1. Build processed data

```bash
python -m data_utils.waymo_sim2real.preprocess \
  --input /mnt/disk/data/public/waymo/perception_1_4_3/training \
  --output artifacts/waymo_sim2real/full/training/processed \
  --workers 8 --resume
```

Useful smoke-test flags are `--max-segments 1` and
`--max-frames-per-segment 8`. Each processed NPZ contains only:

- RGB images for `front`, `front_left`, and `front_right`, resized to 384x256;
- drawable agent boxes and road segments in the ego box-center frame;
- the ego row and exact three-camera raster rig;
- original camera calibration and frame identity for geometry audits.

It does not retain lidar range images, 2D labels, unused side cameras, human
trajectories, or the raw protobuf payload.

## 2. Extract frozen teacher features

```bash
python -m data_utils.waymo_sim2real.extract_teacher_features \
  --processed artifacts/waymo_sim2real/full/training/processed \
  --checkpoint experiments/puffer_drive_cam_gwvaxkmh/model_puffer_drive_cam_007800.pt \
  --output artifacts/waymo_sim2real/full/training/teacher_features \
  --batch-size 128 --loader-workers 8 --resume
```

Every output NPZ contains the frozen 256-D scene feature and the three 96x64
sim renders. The manifest pins the teacher checkpoint by SHA256 so cached targets
cannot silently be mixed across teacher versions.

## 3. Plot the paired sample

```bash
python -m data_utils.waymo_sim2real.visualize \
  --processed artifacts/waymo_sim2real/full/training/processed \
  --features artifacts/waymo_sim2real/full/training/teacher_features \
  --output-dir artifacts/waymo_sim2real/full/training/png \
  --workers 8 --resume
```

Columns are front-left, front, front-right. Rows are sim render, Waymo real, and
a 50% alpha overlay. Bulk PNGs are grouped into one subdirectory per segment.
Use `--output path/to/one.png` instead of `--output-dir` for a single preview.

All three commands support interruption-safe `--resume`; `--overwrite` is the
explicit replacement mode. Run the same commands with `validation` substituted
for `training` to build the validation split.

## 4. Verify a complete artifact tree

```bash
python -m data_utils.waymo_sim2real.verify \
  --root artifacts/waymo_sim2real/full --workers 16
```

The verifier checks one-to-one manifest/file mappings, every teacher feature
and checkpoint hash, every PNG header, and fully decodes one processed sample
and PNG per segment.

## 5. Reconstruct the ego state

The planning-loss term feeds the frozen policy head the same 11-D ego vector the
simulator writes in `compute_observations`. Waymo never labels the ego vehicle,
so it is rebuilt from the per-frame pose in a second pass over the raw records.
This pass decodes only `timestamp_micros` and `pose` -- no JPEG, no lidar -- and
runs at roughly a segment per five seconds per worker.

```bash
for split in training validation; do
  python -m data_utils.waymo_sim2real.extract_ego_state \
    --input /mnt/disk/data/public/waymo/perception_1_4_3/$split \
    --output artifacts/waymo_sim2real/full/$split/ego_state \
    --workers 8 --resume
done
```

Speed, acceleration, yaw rate and steering come from finite differences of the
box-centre pose track and are clipped to the ranges the JERK integrator can
reach, so the head is never asked about a state self-play could not produce.
The goal is the ego's own logged pose 30 m further along its path, matching the
simulator's `goal_target_distance`; segments that never travel that far are
extended along the final heading rather than collapsing the goal onto the
vehicle. `verify.py` checks the join whenever `ego_state/` exists.

## 6. Distill real perception

Install the torchvision build matching this repository's PyTorch/CUDA pair and
cache the official ImageNet weights once:

```bash
uv pip install --python .venv/bin/python torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121 --no-deps
```

```bash
wandb login

.venv/bin/python -m data_utils.waymo_sim2real.train_distillation \
  --root artifacts/waymo_sim2real/full \
  --checkpoint experiments/puffer_drive_cam_gwvaxkmh/model_puffer_drive_cam_007800.pt \
  --output artifacts/waymo_sim2real/runs/real_perception_convnext_tiny \
  --batch-size 32 --workers 8 --amp bf16 \
  --learning-rate 3e-4 --backbone-learning-rate 3e-5 \
  --wandb-name real_perception_convnext_tiny
```

### The objective

This is the sim-to-real stage of Rowe et al.'s Gigapixel recipe (arXiv
2606.19641, Sec. 3.3) applied to `DriveCam`:

    L = feature_weight * ||E_real - E_sim||^2 + cosine_weight * (1 - cos) + plan_weight * KL_plan

Two copies of the pinned policy share one **frozen** planning head -- the ego
encoder, trunk and actor -- so the only thing the optimizer touches is the
student's image backbone. `KL_plan` is the divergence between that head's
action distributions on the teacher's cached feature and on the student's
prediction; `plan_agreement` reports how often their argmax actions match.

The feature term alone weights every latent direction by its coordinate scale.
The planner weights each direction by how much it actually moves the wheel,
which is why the paper's ablation loses 14.7 HD-Score points without it and 2.7
more when the head is unfrozen. `--plan-weight 0` drops the term and the
ego-state requirement, which is useful only as the matching ablation.

The simulation network is never trained: targets are the cached frozen features,
and both splits, every target file and `--checkpoint` must carry the same
teacher SHA256.

### Reading the metrics

`teacher_features/target_stats.json` caches the mean and variance of the teacher
features, and `run.json` prints the variance as `target_variance`. That number is
what a constant predictor scores, and it is the only honest reference here: these
features carry a large mean offset, so predicting the training mean already
reaches a cosine similarity around 0.77. Read `r2` (variance explained), never
`cosine`, as the headline.

`r2_within_segment` removes each segment's own mean from both sides. Roughly half
of the feature variance is which segment a frame came from -- the static look of
the street, which a student can match without resolving a single vehicle. The
within-segment number is the half that perception is actually for.

### Flags that matter

- `--train-frame-stride N` keeps every Nth frame of each training segment. Waymo
  logs at 10 Hz and neighbouring frames differ by about 1% of the feature
  variance, so `5` costs almost no information and makes epochs five times
  cheaper. The scene count, not the frame count, is the real dataset size.
- `--freeze-backbone-stages N` holds the first N of ConvNeXt's four stages fixed.
  The paired set covers a few hundred distinct scenes however many frames it
  holds, and full fine-tuning of 30M parameters overfits inside one epoch; this
  is the cheap counterpart to the LoRA the paper puts on its own backbone.
- `--standardize-targets` divides the feature residual by each dimension's
  standard deviation so no single high-variance coordinate dominates. The
  prediction handed to the planning head stays unscaled.
- `--amp fp16` now skips overflowing steps the way `GradScaler` intends and
  reports them as `skipped_steps`; under `bf16`/`off` a non-finite gradient is
  still a hard error, because there is no loss scale to retune.

### Run directories

Training writes `run.json`, append-only `metrics.jsonl`, and interruption-safe
`last.pt`/`best.pt`. Starting a fresh run on top of an existing directory is
refused: it would append a second history to the same metrics file and reset the
best-loss watermark, overwriting a good checkpoint with a worse first epoch. Use
`--resume <run>/last.pt` to continue, `--overwrite` to discard, or a new
`--output`. A resume must keep every schedule-shaping flag identical -- `epochs`,
batch size, accumulation, learning rates, warmup and the frame stride -- because
`LambdaLR` stores only its step counter, not the curve it indexes.

Only mild photometric jitter is applied; no crop, flip, or geometric transform is
used, because those would invalidate the camera-to-simulation pairing.

W&B tracking starts in online mode by default under project
`pufferdrive-sim2real`. It logs rolling train metrics every 50 optimizer steps,
complete train/validation metrics each epoch, both learning rates, gradient
norm, epoch time, and peak GPU memory. Configure it with `--wandb-project`,
`--wandb-entity`, `--wandb-name`, `--wandb-group`, `--wandb-tags`, and
`--wandb-log-interval`. The run ID is saved in both `run.json` and every
checkpoint, so `--resume` reconnects to the same online run. Use
`--wandb-mode offline` when disconnected or `--wandb-mode disabled` to opt out.

For a short launch check before a full run:

```bash
.venv/bin/python -m data_utils.waymo_sim2real.train_distillation \
  --root artifacts/waymo_sim2real/full \
  --checkpoint experiments/puffer_drive_cam_gwvaxkmh/model_puffer_drive_cam_007800.pt \
  --output /tmp/waymo_distill_smoke --epochs 1 --batch-size 4 --workers 0 \
  --max-train-samples 16 --max-validation-samples 16 --wandb-mode offline
```
