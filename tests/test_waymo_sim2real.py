from pathlib import Path
import json
import struct

import numpy as np
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
from data_utils.waymo_sim2real.real_perception import (
    FeatureAlignmentLoss,
    RealPerception,
    RealPerceptionConfig,
)
from data_utils.waymo_sim2real.train_distillation import PairedWaymoFeatureDataset
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
    loss, metrics = FeatureAlignmentLoss(cosine_weight=0.1)(prediction, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss", "mse", "rmse", "cosine"}
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_feature_alignment_is_zero_for_identical_nonzero_features():
    target = torch.randn(4, 256)
    loss, metrics = FeatureAlignmentLoss()(target, target)
    torch.testing.assert_close(loss, torch.zeros(()), atol=1e-7, rtol=0)
    torch.testing.assert_close(metrics["cosine"], torch.ones(()), atol=1e-6, rtol=0)


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
                "samples": [{"file": sample_name, "processed_file": sample_name}],
            }
        )
    )
    dataset = PairedWaymoFeatureDataset(split)
    images, target = dataset[0]
    assert images.shape == (3, 3, 256, 384)
    assert images.dtype == torch.uint8
    torch.testing.assert_close(target, torch.arange(256, dtype=torch.float32))
    assert dataset.teacher_checkpoint_sha256 == teacher_hash
