# Waymo sim-to-real paired-data pipeline

The pipeline has a strict raw-data boundary:

1. `preprocess.py` is the only stage that reads Waymo Perception TFRecords.
2. `extract_teacher_features.py` reads compact processed NPZ samples, renders
   their abstract scene, and extracts the frozen 256-D `DriveCam` scene feature.
3. `visualize.py` reads the two derived datasets and makes a 3x3 audit PNG.

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

## 5. Distill real perception

Install the torchvision build matching this repository's PyTorch/CUDA pair and
cache the official ImageNet weights once:

```bash
uv pip install --python .venv/bin/python torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121 --no-deps
```

```bash
.venv/bin/python -m data_utils.waymo_sim2real.train_distillation \
  --root artifacts/waymo_sim2real/full \
  --output artifacts/waymo_sim2real/runs/real_perception_convnext_tiny \
  --batch-size 8 --accumulation-steps 4 --workers 8 --amp bf16 \
  --learning-rate 3e-4 --backbone-learning-rate 3e-5
```

The student uses the official torchvision ConvNeXt-Tiny
`IMAGENET1K_V1` backbone shared across the three 384x256 cameras. Its original
classifier is discarded and a new camera-aware fusion head maps the pooled
views into the teacher's 256-D scene space. This is deliberately the simple
global-pooling baseline; it does not contain an FPN or detector. The backbone
uses one tenth of the new head's learning rate. Pass `--random-init` only for a
from-scratch ablation.

The simulation network is not instantiated during training: targets are the
cached frozen features, and both splits plus every target file must carry the
same teacher checkpoint SHA256. This prevents the simulation perception weights
from being updated accidentally.

Training writes `run.json`, append-only `metrics.jsonl`, and interruption-safe
`last.pt`/`best.pt` checkpoints. Resume with `--resume <run>/last.pt`. Only mild
photometric jitter is applied; no crop, flip, or geometric transform is used,
because those would invalidate the camera-to-simulation pairing.

For a short launch check before a full run:

```bash
.venv/bin/python -m data_utils.waymo_sim2real.train_distillation \
  --root artifacts/waymo_sim2real/full \
  --output /tmp/waymo_distill_smoke --epochs 1 --batch-size 1 \
  --accumulation-steps 1 --workers 0 \
  --max-train-samples 2 --max-validation-samples 2
```
