# CARLA sim-to-real paired-data pipeline

A second source of paired `(real image, abstract scene)` samples for the same
distillation the Waymo pipeline feeds. It writes the **same artifact tree**, so
every downstream stage is the Waymo one, run unchanged against a different
`--processed` / `--root`.

```
data_utils/carla_sim2real/collect.py     ->  processed/ + ego_state/
data_utils/carla_sim2real/split_distillation.py  ->  leak-free train/validation
data_utils/waymo_sim2real/extract_teacher_features.py  ->  teacher_features/
data_utils/waymo_sim2real/visualize.py   ->  png/
data_utils/waymo_sim2real/verify.py      ->  checks the tree
data_utils/waymo_sim2real/train_distillation.py  ->  trains the student
```

## Why

`artifacts/waymo_sim2real/runs/distill_v2` is data-bound, not capacity-bound:
validation `r2` plateaus at 0.53 by epoch 8 while train climbs to 0.94, with
`--freeze-backbone-stages 2` already on. Waymo Perception training is 798
segments, and the README's own note that "the scene count, not the frame count,
is the real dataset size" is what that gap measures.

CARLA adds scenes with exact ground-truth pairing — no ego-pose reconstruction,
no label noise, no map guessing — plus appearance diversity a recorded log
cannot have, since each Waymo segment was captured once under one sky.

It does **not** replace the Waymo set. Three towns is far less layout diversity
than 798 real road networks, and a student trained on CARLA pixels still has a
CARLA-to-Waymo gap. This is a pretraining/regularization source.

## Setup

The CARLA Python API must match this venv's Python 3.12. A stale 0.9.13 egg on
`PYTHONPATH` shadows it and fails with `undefined symbol: _Py_tracemalloc_config`,
so unset the variable for these commands.

```bash
uv pip install --python .venv/bin/python \
  /home/tjhu78u/CARLA_0_9_16/PythonAPI/carla/dist/carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl

env -u PYTHONPATH .venv/bin/python -c "import carla; print(carla.__file__)"
```

Start the server (needs ~3-4 GiB of VRAM; check `nvidia-smi` first if a training
job is sharing the card):

```bash
/home/tjhu78u/CARLA_0_9_16/CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000
```

## 1. Collect

```bash
env -u PYTHONPATH .venv/bin/python -m data_utils.carla_sim2real.collect \
  --output artifacts/carla_sim2real/training \
  --town Town01 --town Town02 --town Town10HD \
  --episodes 300 --vehicles 60 --walkers 30 --seed 0 --resume
```

One segment is **91 frames at 10 Hz**, matching `episode_length = 91` and
`dt = 0.1` in `config/ocean/drive_3cam.ini`. Traffic Manager drives the ego and
all traffic; this process only observes, which is the direct analogue of Waymo
log replay.

### Frames, segments, spawns

Three nested loops, and only the middle one is a scene:

```
load_world(town)              once per town, not per segment
└── spawn ego + traffic       --segments-per-spawn segments come from one spawn
    ├── warm-up               --warmup ticks, discarded
    └── segment               --segment-frames ticks, all simulated
        └── written frame     every --frame-stride-th tick -> one .npz
```

**Adjacent frames are nearly redundant.** 91 frames at 10 Hz is 9.1 s of one
scene, and the distillation is bound by *scene* count, not frame count -- that is
what the 0.94 train / 0.53 validation gap measures. `--frame-stride 10` writes
10 frames per segment instead of 91, so a fixed disk budget buys nine times the
scenes.

Striding thins **writes only**. Every tick is still simulated, because the ego
track has to stay at 10 Hz for the finite differences behind `obs[6..8]` to mean
what the planning head expects: a 5-frame brake pulse between two kept frames
saturates `obs[7]` at 0.625 when computed at 10 Hz and reads 0.1875 -- on the
wrong frame -- if rebuilt from the thinned track. `episode_ego_obs` therefore
runs on the full-rate track and the result is row-selected afterwards.

Segments from one spawn are **not** independent, and a stopped ego makes them
identical: at a red light the next window opens exactly where the last one did.
Measured on Town10HD, two such first frames differ by 4.0/255 -- nine times
*less* than two frames one second apart inside a moving segment. So
`--min-segment-displacement` (default 25 m) drops a segment whose ego has not
moved that far since the last *kept* one. Displacement accumulates from the last
kept segment, not the previous one, so the window where the car arrives at the
light survives and only its copies are dropped; stopped traffic is about a fifth
of Waymo's training set and deleting it would bias the data away from exactly
the case the goal term exercises. On the run above it took the most-similar pair
from 4.0/255 to 33.1/255 while leaving the median at 59.

Because dedup discards windows you already paid ticks for, `--segments-per-spawn`
only pays when the ego keeps moving. Measured cost per kept scene, 8 requested:

| | `--segments-per-spawn 1` | `--segments-per-spawn 4` |
|---|---|---|
| Town01 | 121 ticks | 158 ticks (5 of 8 kept) |
| Town10HD | 121 ticks | 98 ticks (8 of 8 kept) |

The default of 1 is the predictable choice at a flat 121 ticks per scene; raising
it is a bet on free-flowing traffic.

**Do not raise `--segment-frames` to get a longer drive.** The segment *is* the
goal window: the frozen head trained on a goal decaying to zero over exactly 91
steps, so a 910-frame segment starts `obs[0]` about ten times too high, off the
manifold the head ever saw. Use `--segments-per-spawn`, which records
consecutive 91-step windows from one spawn -- a longer continuous drive, the
goal distribution intact, and the 30-tick warm-up amortised across all of them.
Each segment is gated independently, so one bad window is dropped without
discarding the good ones beside it.

### Cameras render only around kept frames

A stopped camera is not rendered at all: measured 63.1 ms per tick with the rig
listening against 28.0 ms stopped, the whole 35 ms being the three captures. At
`--frame-stride 10` nine ticks in ten need no image, so the collector subscribes
just before a kept frame and unsubscribes right after. `_await_frame` still
asserts `image.frame == frame_id`, so a mistimed toggle fails loudly rather than
pairing an abstract scene with a neighbouring image.

This is **not** free. Unreal's temporal anti-aliasing accumulates across frames,
so a capture taken cold differs from the same frame in a continuously rendered
run. `--camera-prime` (default 2) starts rendering that many ticks early to give
TAA some history, but it does not fully reconverge. Measured against an
always-on run of the same seed, whose geometry is bit-identical (agents, roads
and `ego_obs` all match exactly):

| `--camera-prime` | mean pixel diff vs always-on | excess high-frequency energy |
|---|---|---|
| 0 | 6.31/255 | +6.4% |
| 1 | 5.33/255 | +7.9% |
| 2 | 4.90/255 | +5.9% |
| 3 | 4.00/255 | +5.3% |

The residual is edge aliasing on building silhouettes and foliage. It is uniform
across the dataset, and small next to the CARLA-to-Waymo appearance gap the
student has to cross anyway — but it is a real difference, so `--frame-stride 1`
(which renders every tick and never toggles) remains the way to get renders
identical to a continuous run.

### Throughput

Measured on an idle A6000, Town10HD, 50 vehicles + 25 walkers + 3 cameras at
`-quality-level=Epic`: **~87 ms per simulated tick**. Of an isolated tick,
24.8 ms is physics and Traffic Manager, 35 ms is the three camera renders.
Striding does not reduce tick cost -- the renders still happen -- so it trades
wall-clock for scenes, not against them.

Toggling the cameras cuts the same 788-tick workload from 68.5 s to 51.0 s
end to end -- 1.34x, or 1.46x on the simulation once the ~13 s world load is
excluded. It is not the 2x the raw render share suggests, because priming pays
three renders per kept frame rather than one.

A 50k-sample run at `--frame-stride 10 --camera-prime 2`, one segment per spawn:

| | |
|---|---|
| segments (scenes) | 5,000 |
| simulated ticks | 5,000 x 121 = 605,000 |
| camera renders | 5,000 x 30 = 150,000 (of 605,000 ticks) |
| wall clock | ~10 h (~15 h rendering every tick) |
| disk | ~28 GB (0.56 MB/sample) |

The same 50k samples at `--frame-stride 1` costs ~2 h and 26 GB but yields
**550** scenes rather than 5,000. Waymo Perception training is 798.

Each episode gets its own sky (`Weather.sample`), skewed toward daylight because
Waymo Perception is overwhelmingly daytime — the goal is to widen appearance
around that distribution, not replace it. True night is excluded: the cameras
would carry almost no signal while the abstract render is unchanged, which is an
unlearnable pair. The full weather is recorded per segment so it can be
conditioned on or ablated later.

`--resume` skips segments already in the manifest and reproduces the sky the
skipped slot would have had, because the per-episode seed is a CRC of
`(seed, split-tag, town, episode)` rather than a salted `hash()`.

### Towns

`Town01`, `Town02` and `Town10HD` only, unless `--allow-elevated-towns` is
passed. The rasterizer is a **flat-ground model**: roads render at `z = 0` and
agent boxes sit on `z = 0`. Town03 (multi-level), Town04 (highway grades) and
Town05 have real elevation, and pairs from them will misalign vertically until a
`|Δz|` frame filter exists.

Town01 and Town02 have no crosswalk records in their OpenDRIVE; Town10HD has 64
crosswalk segments. None of the CARLA towns carry speed bumps.

### Lane width

CARLA's lanes are 3.5-4.0 m, but nothing here converts them. `town_roads` emits
ROAD_LANE **centrelines**, and `render_roads.prepare_runtime_roads` — the same
function the Waymo path calls, mirroring `fill_render_roads` in `drive.h` —
widens them into 4.5 m strips at render time. Waymo's 3.2-3.7 m lanes get the
identical treatment, so the bias is *consistent across domains*, which is what
distillation needs.

## 2. Build the 1k distillation split

The checked 1k sample contains 102 complete ten-frame segments, with 34 from
each of Town01, Town02, and Town10HD. Split whole segments, not frames, so no
near-identical temporal neighbours leak into validation:

```bash
env -u PYTHONPATH .venv/bin/python -m data_utils.carla_sim2real.split_distillation \
  --source artifacts/carla_sim2real/sample1k/training \
  --output artifacts/carla_sim2real/sample1k_dino \
  --train-per-town 27 --validation-per-town 7 --seed 42
```

This produces 810 training frames from 81 segments and 210 validation frames
from 21 segments. Processed images and ego tables are hardlinked, so the split
does not duplicate the camera data.

## 3. Refresh the requested giga teacher targets

The requested teacher is:

```bash
CKPT=experiments/skynet/model_puffer_giga_3cam_001400.pt
ROOT=artifacts/carla_sim2real/sample1k_dino

for split in training validation; do
  env -u PYTHONPATH .venv/bin/python -m data_utils.waymo_sim2real.extract_teacher_features \
    --processed $ROOT/$split/processed --checkpoint $CKPT \
    --output $ROOT/$split/teacher_features \
    --reuse-sim-images artifacts/carla_sim2real/sample1k/training/teacher_features
done
```

`--reuse-sim-images` reuses the already audited abstract renders while
recomputing the 256-D `scene_encoder` target with the new checkpoint. The
feature manifest and every sample pin the teacher SHA256.

## 4. Train the DINOv2 student

The default student matches DrivoR's useful ingredients: a pretrained
DINOv2-S/14 register model shared over three cameras, 16 learned scene tokens
per camera, and rank-32 LoRA on Q and V in every attention block. Its frozen
base has 21,735,936 parameters; LoRA, registers, two-layer fusion transformer,
and 256-D projection contribute 4,256,512 trainable parameters. The simulation
teacher's visual encoder has 896,544 parameters.

```bash
env -u PYTHONPATH .venv/bin/python -m data_utils.waymo_sim2real.train_distillation \
  --root artifacts/carla_sim2real/sample1k_dino \
  --checkpoint experiments/skynet/model_puffer_giga_3cam_001400.pt \
  --output artifacts/carla_sim2real/runs/dino_carla1k_clean \
  --backbone-weights artifacts/carla_sim2real/weights/dinov2_vits14_reg4/model.safetensors \
  --backbone-revision c04b5193082a8d5b0c4856c7937384a48136c5de \
  --epochs 10 --batch-size 4 --accumulation-steps 8 --workers 4 \
  --amp bf16 --wandb-mode disabled
```

The giga planner expects 24 ego inputs: the existing 11-D vehicle state plus
13 training-time conditioning values. Since recorded CARLA has no corresponding
domain parameters, the trainer deterministically samples one valid normalized
conditioning vector per segment. Changing `--conditioning-seed` changes this
assignment; it is fixed across all frames and resumes for a given segment.

The planning KL is not a feed-forward approximation. Both teacher and student
latents pass through the frozen ego encoder, trunk, exact checkpoint LSTMCell
with `h0 = c0 = 0`, and actor head. This implements the first-frame/no-memory
assumption while keeping the planner frozen.

The best epoch also writes a self-contained `deployment.pt`. Load it with
`load_deployment_bundle` from `data_utils.waymo_sim2real.real_perception`; it
returns an image encoder with input `[B, 3, 3, 256, 384]` and output `[B, 256]`.

## 5. Reuse the same stages for Waymo

The trainer reads only the shared `processed/`, `teacher_features/`, and
`ego_state/` contract. Point `--root` at a tree under
`artifacts/waymo_sim2real` after extracting targets with the same teacher; no
model or dataset code changes are required. CARLA remains a pretraining source,
while Waymo supplies the real-domain fine-tuning data.

## Legacy extraction and audit commands

```bash
CKPT=experiments/skynet/model_puffer_giga_3cam_001400.pt
ROOT=artifacts/carla_sim2real

env -u PYTHONPATH .venv/bin/python -m data_utils.waymo_sim2real.extract_teacher_features \
  --processed $ROOT/training/processed --checkpoint $CKPT \
  --output $ROOT/training/teacher_features --batch-size 128 --loader-workers 8 --resume

env -u PYTHONPATH .venv/bin/python -m data_utils.waymo_sim2real.visualize \
  --processed $ROOT/training/processed --output-dir $ROOT/training/png --resume

env -u PYTHONPATH .venv/bin/python -m data_utils.waymo_sim2real.verify --root $ROOT --workers 16
```

These generic stages remain useful for a larger collected tree. Mixing CARLA
and Waymo in one minibatch would need a concat dataset; staged CARLA pretraining
followed by Waymo fine-tuning needs no format conversion.

## The geometry

### Camera rig

The rig is not designed here. It is `raster_ref.WAYMO_RIG`, and the CARLA
sensors are placed to match it, so the abstract render and the captured image
are one pinhole camera at two resolutions:

| | render | CARLA image |
|---|---|---|
| resolution | 96x64 | 384x256 |
| fx = fy | 103.335 | 413.34 |
| cx, cy | 48, 32 | 192, 128 |
| horizontal FOV | 49.8306 deg | 49.8306 deg |

`Camera.intrinsics()` is a symmetric pinhole and so is CARLA's RGB camera, so
passing that FOV at 384x256 reproduces the render intrinsics scaled by exactly
four. `tests/test_carla_sim2real.py` asserts the round trip.

### Frames

CARLA's world is left-handed with y pointing right; OpenDRIVE and this
repository are right-handed. Every world quantity negates y, and yaw with it.
`roads.world_to_ego` is the only place that convention is applied.

Camera mounts additionally negate yaw and pitch (`Camera` documents positive
pitch as tilting the view *down*; CARLA's tilts it *up*) and offset by the
actor's `bounding_box.location` in x and y — but **not** z, because
`camera.pos[2]` and the CARLA actor origin are both measured from the road.

**A yaw-sign error here silently swaps `front_left` and `front_right`.** Run the
calibration audit below before any bulk collection.

### Ego observations

`ego.episode_ego_obs` calls `waymo_sim2real.ego_state.ego_observations` — the
exact code the Waymo branch runs, so any bias in the finite differences is
common to both sources — with two arguments changed:

- `axle_to_center=0.0`: CARLA's actor transform plus `bounding_box.location` is
  already the box centre, so there is no rear-axle offset to undo.
- `goal=`: the episode's own final box centre, held fixed.

That second one matters. `drive_3cam.ini` sets `goal_behavior = 0`, so
`sample_new_goal` (`drive.h:2282`) is **unreachable** and the frozen planning
head trained against a fixed world point read from the map binary
(`drive.h:576`) that decays toward the vehicle across the episode. The Waymo
extractor's rolling 30 m lookahead pins `obs[0]` near 0.15; the endpoint goal
sweeps it 0.45 -> 0 over 91 frames, which is the distribution the head actually
saw.

**This makes the two branches' ego vectors differently distributed**, which
matters for staged pretraining. The `goal=` keyword also exists on the Waymo
path now, so re-extracting `ego_state` there in 91-frame windows would align
them — cheap, no image reprocessing, but it changes the existing baseline.

## Verification

```bash
env -u PYTHONPATH .venv/bin/python -m pytest tests/test_carla_sim2real.py -q
```

All offline — `carla.Map(name, xodr_content)` builds a full road network from an
`.xodr` string in-process, so the coordinate-frame test needs no server. That
test matches `town_roads("Town01")` against the checked-in
`data_utils/carla/carla_py123d/Town01.json`, which came from a different tool in
the right-handed frame: the two agree to a 95th-percentile 0.47 m, and a
mirrored y would blow it up. It is the cheapest possible proof of the convention.

### Rig calibration audit

The offline tests prove the *rig* is right (a point 20 m ahead projects to
u = 47.9 on a 96-px camera whose cx is 48, and off-frame on both yawed cameras).
They cannot prove the CARLA sensors were *mounted* to match it, because that
needs a server.

The audit that does: project known ego-frame points through the rig, project the
same world points through CARLA's own `sensor.get_transform().get_inverse_matrix()`,
and compare. Measured on Town01 with a settled ego:

    in-frame reprojection error: max 0.2864 px, mean 0.1572 px   (image is 384x256)

The residual is fully accounted for — the settled ego rests at world
`z = -0.0053`, putting the front sensor at 2.1104 m against the rig's assumed
2.1157 m. **Let the ego settle before measuring**: a probe that ticks only twice
still has the car falling from its spawn point and shows a ~0.095 m height
offset, which reads as a 2-5 px error that grows as 1/distance.

The `visualize.py` 3-row overlay is the standing check. On CARLA the abstract
render sits almost perfectly on the image — unlike Waymo, where map and label
noise are visible in the overlay.

## Sizing

Waymo averages ~660 KB per sample. 3 towns x 300 episodes x 91 frames is ~82k
samples, roughly 55 GiB, against 655 GiB free on `/mnt/disk`. That is ~900
distinct scenes versus Waymo's 798 — the scene count the `r2` gap says is the
binding constraint.

Road segments per town: Town01 13,735, Town02 6,034, Town10HD 9,922. After the
220 m cull a frame carries roughly 6-9k, against Waymo's ~9.6k.

Measured throughput on an idle A6000 at Epic quality: ~15 s per episode including
the 30 warm-up ticks, so 900 episodes is about 3.5 hours.

### Splits

The per-episode seed folds in `--split-tag`, which defaults to the output
directory's name. Without that, collecting `validation/` with the same `--seed`
as `training/` would reproduce **the same episodes** — CARLA is deterministic
given the seed, so the sky, the Traffic Manager seed and the spawn point would
all repeat. The default keeps the two disjoint without having to remember.

### The Traffic Manager must be torn down between towns

This one is worth stating plainly because nothing downstream can detect it. The
Traffic Manager lives in the *server*, keyed by port, so `get_trafficmanager`
returns the same instance however often the handle is re-fetched. Carried across
a `load_world`, it keeps the previous town's actor registry and its path
following collapses: traffic spawns correctly on the road and, within about five
seconds, drives onto the pavement and jams. Measured on Town01, 50 vehicles,
60 ticks:

    tick   offroad>3m   max roll   fleet median speed
       0        0/50       0.00        0.00 m/s
      20        3/50       5.00        2.97 m/s
      50       27/50     104.53        0.51 m/s
     130       31/50      17.06        0.02 m/s
    final lane types: Sidewalk 31, Shoulder 11, Driving 8

`load_town` therefore calls `shut_down()` on the old manager *before*
`load_world` and re-acquires it after, then sets synchronous mode, seed and
`hybrid_physics_mode(False)` on the fresh one. Hybrid mode is explicitly off:
it freezes physics for vehicles far from a hero actor, which with a single hero
leaves most of the fleet stationary.

The symptom is order-dependent, which makes it easy to misdiagnose — the third
town collected behaves far worse than the same town collected first. Measured
across three towns, vehicles only, share further than 3 m from a lane
centreline: **32.1% before, 0.0% after**.

### What the collector rejects

An ego that gets shoved onto grass by dense spawning still records 91 perfectly
well-paired frames — the abstract render correctly shows black ground with the
road in the distance — so nothing looks broken. It simply is not a driving
scene. Three gates run, and all three are needed:

1. **After the warm-up**, the ego's distance to the nearest lane centreline is
   checked against `--max-offroad` (default 4 m).
2. **After the warm-up**, the share of traffic off the driving lanes is checked
   against `--max-stray-fraction` (default 0.25). A healthy fleet measures 0.00.
3. **After recording**, both are re-checked over the 91 frames actually
   captured. The warm-up gates only prove the episode *started* on-road; the
   ego also drives into buildings mid-episode, and the resulting frames are
   correctly paired — three cameras pressed against a stone wall, with the
   abstract render faithfully showing black ground. Only the recorded track
   reveals it.

A failing episode is respawned up to `--max-attempts` times.

### `episodes.jsonl`

One row per segment at the split root, carrying the sky, the town, the seed, the
ego blueprint and its dimensions, mean speed, mean agents in view, and
`max_offroad` — the worst the ego strayed *during* recording, so mid-episode
excursions stay queryable even though rejection only happens post-warm-up.

Weather is the whole reason this file exists: it is the axis CARLA adds that a
recorded log cannot, and without it the sky is applied and then lost.

### Stationary episodes

The progress line reports mean speed and flags anything under 0.5 m/s as
`<-- stationary`. Some of those are real: a car held at a red light for the full
9.1 s is exactly the case Waymo's own notes call out as a fifth of its training
set. Check the overlay before assuming a bug — but do check, because a pinned ego
still yields 91 perfectly well-formed frames and is otherwise invisible.
