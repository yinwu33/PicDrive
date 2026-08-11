from pathlib import Path
import json
import struct

import numpy as np
import pytest
from PIL import Image
import torch

from data_utils.waymo_sim2real.preprocess import SENSOR_TO_CV, _calibration_arrays
from data_utils.waymo_sim2real.processed import (
    CAMERA_NAMES,
    FEATURE_SCHEMA_VERSION,
    PROCESSED_SCHEMA_VERSION,
    SIM_HEIGHT,
    SIM_WIDTH,
    atomic_savez,
    load_feature,
    load_processed,
    load_render_input,
)
from data_utils.waymo_sim2real.proto import iter_fields, repeated_doubles
from data_utils.waymo_sim2real.ego_state import ego_observations
from data_utils.waymo_sim2real.processed import EGO_OBS_DIM, EGO_SCHEMA_VERSION
from data_utils.waymo_sim2real.real_perception import (
    DistillationLoss,
    RealPerception,
    RealPerceptionConfig,
)
from data_utils.waymo_sim2real.train_distillation import (
    PairedWaymoFeatureDataset,
    _within_segment_r2,
    _wandb_epoch_payload,
)
from data_utils.waymo_sim2real.visualize import plot_sample


def _varint(value):
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def test_proto_reader_accepts_packed_and_unpacked_doubles():
    unpacked = b"".join(_varint((1 << 3) | 1) + struct.pack("<d", v) for v in (1.5, 2.5))
    packed_values = struct.pack("<2d", 3.5, 4.5)
    packed = _varint((1 << 3) | 2) + _varint(len(packed_values)) + packed_values
    message = unpacked + packed + _varint(2 << 3) + _varint(17)
    assert repeated_doubles(message, 1) == [1.5, 2.5, 3.5, 4.5]
    fields = list(iter_fields(message))
    assert (fields[-1].number, fields[-1].wire_type, fields[-1].value) == (2, 0, 17)


def test_waymo_calibration_is_converted_to_raster_cv_frame():
    intrinsic = np.asarray([2000.0, 2100.0, 960.0, 640.0, 0, 0, 0, 0, 0])
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [1.54, 0.1, 2.1]
    calibrations = {
        name: {"intrinsic": intrinsic, "extrinsic": extrinsic, "width": 1920, "height": 1280}
        for name in (1, 2, 3)
    }
    rig, source_intrinsics, source_extrinsics, sizes = _calibration_arrays(calibrations)
    np.testing.assert_allclose(rig[0, :9].reshape(3, 3), SENSOR_TO_CV)
    np.testing.assert_allclose(rig[0, 9:12], [0.1, 0.1, 2.1], atol=1e-6)
    np.testing.assert_allclose(rig[0, 12:16], [100.0, 105.0, 48.0, 32.0])
    assert source_intrinsics.shape == (3, 9)
    assert source_extrinsics.shape == (3, 4, 4)
    np.testing.assert_array_equal(sizes, [[1280, 1920]] * 3)


def test_processed_feature_roundtrip_and_visualization(tmp_path: Path):
    processed_path = tmp_path / "processed" / "sample.npz"
    feature_path = tmp_path / "features" / "sample.npz"
    real = np.zeros((3, 32, 48, 3), dtype=np.uint8)
    real[0, ..., 0] = 255
    real[1, ..., 1] = 255
    real[2, ..., 2] = 255
    sim = np.zeros((3, SIM_HEIGHT, SIM_WIDTH, 3), dtype=np.uint8)
    sim[...] = 128
    atomic_savez(
        processed_path,
        schema_version=np.asarray(PROCESSED_SCHEMA_VERSION, dtype=np.int32),
        segment_id=np.asarray("segment"),
        timestamp_micros=np.asarray(123, dtype=np.int64),
        frame_index=np.asarray(0, dtype=np.int32),
        camera_names=np.asarray(CAMERA_NAMES),
        real_images=real,
        agents=np.zeros((0, 8), dtype=np.float32),
        roads=np.zeros((0, 6), dtype=np.float32),
        ego=np.asarray([0, 0, 1, 0, -1], dtype=np.float32),
        rig=np.zeros((3, 20), dtype=np.float32),
        source_intrinsics=np.zeros((3, 9), dtype=np.float64),
        source_extrinsics=np.zeros((3, 4, 4), dtype=np.float64),
        source_image_sizes=np.zeros((3, 2), dtype=np.int32),
    )
    atomic_savez(
        feature_path,
        schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int32),
        segment_id=np.asarray("segment"),
        timestamp_micros=np.asarray(123, dtype=np.int64),
        camera_names=np.asarray(CAMERA_NAMES),
        teacher_feature=np.zeros(256, dtype=np.float32),
        sim_images=sim,
        checkpoint_sha256=np.asarray("0" * 64),
    )
    assert load_processed(processed_path)["real_images"].shape == (3, 32, 48, 3)
    render_input = load_render_input(processed_path)
    assert "real_images" not in render_input
    assert render_input["rig"].shape == (3, 20)
    assert load_feature(feature_path)["teacher_feature"].shape == (256,)

    output = tmp_path / "preview.png"
    plot_sample(processed_path, feature_path, output)
    with Image.open(output) as image:
        assert image.format == "PNG"
        assert image.width > image.height


def test_real_perception_emits_teacher_sized_feature_and_backpropagates():
    config = RealPerceptionConfig(
        fusion_dim=32,
        fusion_dropout=0.0,
    )
    model = RealPerception(config, pretrained=False)
    # The full production backbone is exercised without downloading weights in
    # unit tests. A single minimum-size view keeps the CPU test inexpensive.
    images = torch.randint(0, 256, (1, 3, 3, 32, 32), dtype=torch.uint8)
    prediction = model(images)
    assert prediction.shape == (1, 256)
    assert model.pretrained_source == "random"
    assert model.pretrained_sha256 is None
    target = torch.randn_like(prediction)
    loss, metrics = DistillationLoss(cosine_weight=0.1, plan_weight=0.0)(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "feature_loss", "mse", "cosine"}
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_feature_alignment_is_zero_for_identical_nonzero_features():
    target = torch.randn(4, 256)
    loss, metrics = DistillationLoss(plan_weight=0.0)(target, target)
    torch.testing.assert_close(loss, torch.zeros(()), atol=1e-7, rtol=0)
    torch.testing.assert_close(metrics["cosine"], torch.ones(()), atol=1e-6, rtol=0)


class _ConstantPlanHead(torch.nn.Module):
    """Stand-in for the frozen DriveCam head: linear in scene, biased by ego."""

    def __init__(self, scene_dim=256, ego_features=EGO_OBS_DIM, actions=12):
        super().__init__()
        self.scene_dim = scene_dim
        self.ego_features = ego_features
        self.action_dims = [actions]
        self.linear = torch.nn.Linear(scene_dim + ego_features, actions)
        self.requires_grad_(False)

    def forward(self, scene, ego):
        return [self.linear(torch.cat([scene, ego], dim=1))]


def test_planning_loss_vanishes_on_matching_features_and_never_moves_the_head():
    head = _ConstantPlanHead()
    criterion = DistillationLoss(feature_weight=0.0, cosine_weight=0.0, plan_weight=1.0, plan_head=head)
    target = torch.randn(8, 256)
    ego = torch.randn(8, EGO_OBS_DIM)

    matched, metrics = criterion(target, ego=ego, target=target)
    torch.testing.assert_close(matched, torch.zeros(()), atol=1e-6, rtol=0)
    torch.testing.assert_close(metrics["plan_agreement"], torch.ones(()), atol=0, rtol=0)

    prediction = (target + torch.randn_like(target)).requires_grad_(True)
    diverged, metrics = criterion(prediction, target, ego)
    assert float(diverged) > 0.0 and float(metrics["plan_kl"]) > 0.0
    diverged.backward()
    # The gradient reaches the student and stops at the frozen planner.
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert all(parameter.grad is None for parameter in head.parameters())


def test_planning_loss_requires_ego_and_rejects_a_missing_head():
    with pytest.raises(ValueError, match="frozen planning head"):
        DistillationLoss(plan_weight=1.0)
    criterion = DistillationLoss(plan_weight=1.0, plan_head=_ConstantPlanHead())
    with pytest.raises(ValueError, match="ego observation"):
        criterion(torch.randn(2, 256), torch.randn(2, 256))


def test_target_standardization_reweights_the_residual():
    scale = torch.cat([torch.full((128,), 0.5), torch.full((128,), 2.0)])
    target = torch.zeros(4, 256)
    prediction = torch.ones(4, 256)
    plain, _ = DistillationLoss(cosine_weight=0.0, plan_weight=0.0)(prediction, target)
    scaled, _ = DistillationLoss(cosine_weight=0.0, plan_weight=0.0, target_scale=scale)(
        prediction, target
    )
    # 128 dims divided by 0.5 and 128 by 2.0: 128*4 + 128*0.25 against 256*1.
    torch.testing.assert_close(plain, torch.tensor(256.0))
    torch.testing.assert_close(scaled, torch.tensor(128 * 4.0 + 128 * 0.25))


def test_within_segment_r2_ignores_a_constant_per_segment_offset():
    rng = np.random.default_rng(0)
    targets = rng.normal(size=(40, 8))
    segments = np.repeat([0, 1, 2, 3], 10)
    offsets = rng.normal(size=(4, 8))[segments]
    # A student that nails the dynamics but is biased per segment still scores 1.
    perfect, variance = _within_segment_r2(targets + offsets, targets, segments)
    assert perfect == pytest.approx(1.0, abs=1e-9)
    assert variance > 0.0
    # One that only knows which segment it is in explains none of it.
    segment_means = np.stack([targets[segments == s].mean(axis=0) for s in range(4)])
    blind, _ = _within_segment_r2(segment_means[segments], targets, segments)
    assert blind == pytest.approx(0.0, abs=1e-9)


def test_ego_observation_matches_the_simulator_layout():
    steps = 60
    times = np.arange(steps) * 100_000
    poses = np.tile(np.eye(4), (steps, 1, 1))
    poses[:, 0, 3] = np.arange(steps) * 1.2  # 12 m/s straight down +x
    obs = ego_observations(poses, times)
    assert obs.shape == (steps, EGO_OBS_DIM) and obs.dtype == np.float32
    # Goal sits goal_distance ahead on the driven path, in the ego frame.
    assert obs[0, 0] / 0.005 == pytest.approx(30.0, abs=1e-3)
    assert obs[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert obs[10, 2] * 100.0 == pytest.approx(12.0, abs=1e-3)
    assert obs[:, 5].max() == 0.0 and obs[:, 9].max() == 0.0
    assert obs[:, 10] == pytest.approx(1.0 / 3.0)
    # A straight line has no steering and no lateral acceleration.
    assert abs(obs[:, 6]).max() == pytest.approx(0.0, abs=1e-6)
    assert abs(obs[:, 8]).max() == pytest.approx(0.0, abs=1e-6)


def test_ego_observation_extrapolates_a_goal_for_a_stationary_log():
    times = np.arange(20) * 100_000
    poses = np.tile(np.eye(4), (20, 1, 1))
    obs = ego_observations(poses, times)
    # Waiting at a light must not collapse the goal onto the vehicle.
    assert obs[0, 0] / 0.005 == pytest.approx(30.0, abs=1e-3)
    assert abs(obs[:, 2]).max() == pytest.approx(0.0, abs=1e-9)


def test_paired_distillation_dataset_uses_manifest_mapping(tmp_path: Path):
    split = tmp_path / "training"
    processed_dir = split / "processed"
    feature_dir = split / "teacher_features"
    processed_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    sample_name = "frame.npz"
    teacher_hash = "a" * 64
    atomic_savez(
        processed_dir / sample_name,
        real_images=np.zeros((3, 256, 384, 3), dtype=np.uint8),
    )
    atomic_savez(
        feature_dir / sample_name,
        teacher_feature=np.arange(256, dtype=np.float32),
        checkpoint_sha256=np.asarray(teacher_hash),
    )
    (processed_dir / "manifest.jsonl").write_text(json.dumps({"file": sample_name}) + "\n")
    (feature_dir / "manifest.json").write_text(
        json.dumps(
            {
                "camera_names": list(CAMERA_NAMES),
                "checkpoint_sha256": teacher_hash,
                "feature_dim": 256,
                "samples": [
                    {
                        "file": sample_name,
                        "processed_file": sample_name,
                        "segment_id": "seg",
                        "timestamp_micros": 0,
                    }
                ],
            }
        )
    )
    dataset = PairedWaymoFeatureDataset(split, require_ego=False)
    images, target, ego, segment = dataset[0]
    assert images.shape == (3, 3, 256, 384)
    assert images.dtype == torch.uint8
    torch.testing.assert_close(target, torch.arange(256, dtype=torch.float32))
    assert dataset.teacher_checkpoint_sha256 == teacher_hash
    assert ego.shape == (EGO_OBS_DIM,) and segment == 0


def _write_split(split: Path, segments: dict[str, list[int]], teacher_hash: str, with_ego: bool):
    processed_dir = split / "processed"
    feature_dir = split / "teacher_features"
    processed_dir.mkdir(parents=True)
    feature_dir.mkdir(parents=True)
    processed_lines, samples = [], []
    for segment, stamps in segments.items():
        for stamp in stamps:
            name = f"{segment}__{stamp}.npz"
            atomic_savez(processed_dir / name, real_images=np.zeros((3, 256, 384, 3), dtype=np.uint8))
            atomic_savez(
                feature_dir / name,
                teacher_feature=np.full(256, float(stamp), dtype=np.float32),
                checkpoint_sha256=np.asarray(teacher_hash),
            )
            processed_lines.append(json.dumps({"file": name}))
            samples.append(
                {
                    "file": name,
                    "processed_file": name,
                    "segment_id": segment,
                    "timestamp_micros": stamp,
                }
            )
    (processed_dir / "manifest.jsonl").write_text("\n".join(processed_lines) + "\n")
    (feature_dir / "manifest.json").write_text(
        json.dumps(
            {
                "camera_names": list(CAMERA_NAMES),
                "checkpoint_sha256": teacher_hash,
                "feature_dim": 256,
                "samples": samples,
            }
        )
    )
    if with_ego:
        ego_dir = split / "ego_state"
        ego_dir.mkdir(parents=True)
        for segment, stamps in segments.items():
            atomic_savez(
                ego_dir / f"{segment}.npz",
                schema_version=np.asarray(EGO_SCHEMA_VERSION, dtype=np.int32),
                segment_id=np.asarray(segment),
                timestamp_micros=np.asarray(stamps, dtype=np.int64),
                ego_obs=np.tile(
                    np.arange(EGO_OBS_DIM, dtype=np.float32), (len(stamps), 1)
                ) * np.asarray(stamps, dtype=np.float32)[:, None],
            )


def test_frame_stride_subsamples_inside_each_segment(tmp_path: Path):
    split = tmp_path / "training"
    _write_split(split, {"a": [0, 1, 2, 3], "b": [0, 1, 2]}, "b" * 64, with_ego=False)
    dataset = PairedWaymoFeatureDataset(split, frame_stride=2, require_ego=False)
    # Striding must restart per segment, not run across the concatenated list.
    kept = sorted(path.name for path, _ in dataset.pairs)
    assert kept == ["a__0.npz", "a__2.npz", "b__0.npz", "b__2.npz"]
    assert sorted(set(dataset.segments.tolist())) == [0, 1]


def test_ego_state_is_joined_by_segment_and_timestamp(tmp_path: Path):
    split = tmp_path / "training"
    _write_split(split, {"a": [10, 20], "b": [30]}, "c" * 64, with_ego=True)
    dataset = PairedWaymoFeatureDataset(split, require_ego=True)
    rows = {path.name: dataset.ego[index] for index, (path, _) in enumerate(dataset.pairs)}
    np.testing.assert_allclose(rows["a__20.npz"], np.arange(EGO_OBS_DIM) * 20.0)
    np.testing.assert_allclose(rows["b__30.npz"], np.arange(EGO_OBS_DIM) * 30.0)


def test_missing_ego_state_names_the_command_that_builds_it(tmp_path: Path):
    split = tmp_path / "training"
    _write_split(split, {"a": [0]}, "d" * 64, with_ego=False)
    with pytest.raises(FileNotFoundError, match="extract_ego_state"):
        PairedWaymoFeatureDataset(split, require_ego=True)


def test_wandb_epoch_payload_flattens_local_metrics():
    record = {
        "epoch": 2,
        "global_step": 123,
        "elapsed_seconds": 4.5,
        "best_validation_loss": 0.2,
        "peak_gpu_memory_gib": 3.0,
        "train": {"loss": 0.3, "cosine": 0.7},
        "validation": {"loss": 0.2, "cosine": 0.8},
    }
    assert _wandb_epoch_payload(record) == {
        "epoch": 2.0,
        "epoch/elapsed_seconds": 4.5,
        "validation/best_loss": 0.2,
        "system/peak_gpu_memory_gib": 3.0,
        "train/loss": 0.3,
        "train/cosine": 0.7,
        "validation/loss": 0.2,
        "validation/cosine": 0.8,
    }
