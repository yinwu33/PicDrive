# Waymo sim-to-real paired-data pipeline

The pipeline has a strict raw-data boundary:

1. `preprocess.py` and `extract_ego_state.py` are the only stages that read
   Waymo Perception TFRecords.
2. `extract_teacher_features.py` reads compact processed NPZ samples, renders
   their abstract scene, and extracts the frozen 256-D `DriveCam` scene feature.
3. `visualize.py` reads the two derived datasets and makes a 3x3 audit PNG.
4. `train_distillation.py` reads only derived artifacts and the pinned teacher
   checkpoint; it never renders and never updates the simulation policy.

## Optional: retain raw camera data without LiDAR

`strip_lidar.py` removes only top-level `Frame.lasers` (protobuf field 5) while
preserving images, calibrations, poses, maps, and the field 6 3D boxes consumed
by `preprocess.py`. It rewrites one TFRecord beside the source, validates all
input and output CRC32C checksums, then atomically replaces that source file.
Both the per-file and overall progress bars include an ETA.

```bash
uv pip install --python .venv/bin/python google-crc32c

# Non-mutating trial on the first sorted segment.
.venv/bin/python -m data_utils.waymo_sim2real.strip_lidar \
  --input /mnt/disk/data/public/waymo/perception_1_4_3/training \
  --dry-run --max-files 1

# Destructive in-place conversion, one validated segment at a time.
.venv/bin/python -m data_utils.waymo_sim2real.strip_lidar \
  --input /mnt/disk/data/public/waymo/perception_1_4_3/training \
  --in-place
```

Until a temporary replacement has fully validated, an interrupted run leaves
the current source untouched. Files atomically replaced before the interruption
are already valid stripped TFRecords, and a rerun safely reports them as
unchanged. A temporary copy needs at most the space of one raw segment; the
original inode cannot be shortened directly because TFRecord lengths and
checksums change when a protobuf field is removed.

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
- drawable agent boxes and canonical road segments in the ego box-center frame,
  including `ROAD_LANE` centerlines;
- the ego row and exact three-camera raster rig;
- original camera calibration and frame identity for geometry audits.

It does not retain lidar range images, 2D labels, unused side cameras, human
trajectories, or the raw protobuf payload.

The 2 Hz dataset uses the same command with `--frame-stride 5` and split-specific
outputs under `artifacts/waymo_sim2real/2hz/{training,validation}/processed`.

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

Before either feature extraction or direct visualization, canonical roads are
converted with the same camera-policy semantics as live ocean/teddy/giga:
4.5-metre overlapping lane-area strips, yellow road edges, painted features
first, and black non-road ground.

## 3. Plot the paired sample

```bash
python -m data_utils.waymo_sim2real.visualize \
  --processed artifacts/waymo_sim2real/full/training/processed \
  --output-dir artifacts/waymo_sim2real/full/training/png \
  --resume
```

Columns are front-left, front, front-right. Rows are sim render, Waymo real, and
a 50% alpha overlay. Bulk PNGs are grouped into one subdirectory per segment.
Use `--output path/to/one.png` instead of `--output-dir` for a single preview.
The sim row is rendered directly from each processed NPZ, so teacher features
are not required. Pass `--features path/to/teacher_features` only to reuse sim
images cached by feature extraction. Pass `--flat` to put every PNG directly in
the output directory instead of grouping them by segment.

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

The default student is DINOv2 ViT-S/14 with 16 learned scene registers per
camera and rank-32 Q/V LoRA. `setup.py` pins the required `timm`, `safetensors`,
and Hugging Face client versions. Cache the pinned weights once, or pass a local
`model.safetensors` using `--backbone-weights`.

```bash
uv pip install --python .venv/bin/python -e .
```

```bash
wandb login

.venv/bin/python -m data_utils.waymo_sim2real.train_distillation \
  --root artifacts/waymo_sim2real/full \
  --checkpoint experiments/skynet/model_puffer_giga_3cam_001400.pt \
  --output artifacts/waymo_sim2real/runs/real_perception_dinov2 \
  --batch-size 4 --accumulation-steps 8 --workers 8 --amp bf16 \
  --wandb-name real_perception_dinov2
```

### The objective

This is the sim-to-real stage of Rowe et al.'s Gigapixel recipe (arXiv
2606.19641, Sec. 3.3) applied to `DriveCam`:

    L = feature_weight * ||E_real - E_sim||^2 + cosine_weight * (1 - cos) + plan_weight * KL_plan

Two copies of the pinned policy share one **frozen** planning head -- the ego
encoder, trunk, LSTMCell, and actor. Each forward uses `h0 = c0 = 0`, making the
objective the exact first-frame/no-memory recurrent policy rather than a
feed-forward approximation. `KL_plan` is the divergence between that head's
action distributions on the teacher's cached feature and on the student's
prediction; `plan_agreement` reports how often their argmax actions match.

The giga checkpoint expects a 24-D ego input rather than the stored 11-D base
state. The remaining 13 normalized domain-conditioning values are drawn
deterministically once per segment from the same valid training distributions;
`--conditioning-seed` pins them across frames and resumes.

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
- `--lora-rank`, `--num-scene-tokens`, and `--fusion-layers` control the DINO
  adapter. The pretrained DINO base stays frozen; only Q/V LoRA, scene tokens,
  fusion, and the final projection train. `--architecture convnext_tiny` and
  `--freeze-backbone-stages` retain the older baseline as an ablation.
- `--standardize-targets` divides the feature residual by each dimension's
  standard deviation so no single high-variance coordinate dominates. The
  prediction handed to the planning head stays unscaled.
- `--amp fp16` now skips overflowing steps the way `GradScaler` intends and
  reports them as `skipped_steps`; under `bf16`/`off` a non-finite gradient is
  still a hard error, because there is no loss scale to retune.

### Run directories

Training writes `run.json`, append-only `metrics.jsonl`, interruption-safe
`last.pt`/`best.pt`, and a self-contained `deployment.pt` whenever validation
improves. Starting a fresh run on top of an existing directory is
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
