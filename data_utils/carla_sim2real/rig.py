"""The CARLA camera rig, pinned to the simulator's own three-camera Waymo rig.

The abstract renderer and the CARLA sensor must be the *same* pinhole camera, or
the student spends capacity absorbing a change of projection on top of the
appearance gap.  So the rig is not designed here: it is read from
``raster_ref.WAYMO_RIG``, and the CARLA sensors are placed to match it.

``Camera.intrinsics()`` is a symmetric pinhole (``cx = width/2``), and CARLA's
RGB camera is too, so the match is exact rather than approximate.  At the
Waymo focal length the render and the captured image differ only by the
resolution ratio: 96x64 against 384x256, a clean factor of four.

Nothing here imports ``carla``; mounts are returned as plain tuples so the whole
module stays importable -- and testable -- without the simulator.
"""

from __future__ import annotations

import math

import numpy as np

from pufferlib.ocean.drive.raster_ref import WAYMO_RIG, Camera, rig_tensor

from . import REAL_HEIGHT, REAL_WIDTH

# Waymo's sensor frame is x-forward/y-left/z-up; the rasterizer uses the CV
# x-right/y-down/z-forward frame.  Same matrix as preprocess.SENSOR_TO_CV, but
# imported from nowhere so this module does not depend on the Waymo reader.
SENSOR_TO_CV = np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64)


def rig_cameras() -> list[Camera]:
    """The rig CARLA reproduces: the simulator's three forward Waymo cameras."""
    return list(WAYMO_RIG)


def rig_array() -> np.ndarray:
    """Return the ``[3, 20]`` rig the CUDA rasterizer reads.

    Constant for every sample, which is exactly what ``_extract_batch`` in
    ``extract_teacher_features`` demands -- it refuses a batch that crosses
    camera calibrations, and one rig never can.
    """
    return np.ascontiguousarray(rig_tensor(rig_cameras()).numpy(), dtype=np.float32)


def sensor_fov_deg(camera: Camera) -> float:
    """Horizontal FOV to give CARLA's ``fov`` blueprint attribute, in degrees.

    CARLA builds its projection from the horizontal FOV alone and centres the
    principal point, so passing this angle at any resolution of the same aspect
    ratio reproduces ``camera.intrinsics()`` scaled by the resolution ratio.
    """
    fx, _, cx, _ = camera.intrinsics()
    return 2.0 * math.degrees(math.atan(cx / fx))


def sensor_intrinsics(camera: Camera, width: int = REAL_WIDTH, height: int = REAL_HEIGHT):
    """Return ``(fx, fy, cx, cy)`` of the captured image, not the sim render."""
    fx, fy, cx, cy = camera.intrinsics()
    scale_x, scale_y = width / camera.width, height / camera.height
    return fx * scale_x, fy * scale_y, cx * scale_x, cy * scale_y


def mount(camera: Camera, box_offset=(0.0, 0.0, 0.0)):
    """Return the CARLA attachment ``((x, y, z), (pitch, yaw, roll))`` for a camera.

    Three conventions differ and all three bite:

    * CARLA is left-handed with **y pointing right**; the ego frame here has y
      pointing left, so ``y`` negates.
    * CARLA's positive pitch tilts the view **up**; ``Camera`` documents positive
      pitch as tilting it **down**, so ``pitch`` negates.  Positive CARLA yaw
      turns right, positive ``Camera`` yaw turns left, so ``yaw`` negates too.
    * An attachment is relative to the actor origin, whereas ``camera.pos`` is
      relative to the bounding-box centre.  ``box_offset`` is the actor's
      ``bounding_box.location``, which closes that gap in x and y.  It is
      deliberately *not* applied in z: ``camera.pos[2]`` is measured up from the
      road surface, and so is the CARLA actor origin.

    Roll passes through unchanged.  Every camera in ``WAYMO_RIG`` has zero roll,
    so that leg is untested -- revisit it before adding a rolled camera.
    """
    x, y, z = camera.pos
    return (
        (box_offset[0] + x, box_offset[1] - y, z),
        (-camera.pitch_deg, -camera.yaw_deg, camera.roll_deg),
    )


def source_calibration(width: int = REAL_WIDTH, height: int = REAL_HEIGHT):
    """Return the ``source_intrinsics``/``extrinsics``/``image_sizes`` triple.

    ``validate_processed`` requires these three arrays, and geometry audits read
    them.  The layouts mirror what ``preprocess.py`` stores for Waymo:

    * ``intrinsics [3, 9]`` -- Waymo's ``(f_u, f_v, c_u, c_v, k1, k2, p1, p2, k3)``
      at the *captured* resolution.  CARLA's RGB camera has no lens distortion
      enabled by default, so the five distortion terms are zero.
    * ``extrinsics [3, 4, 4]`` -- sensor-to-ego, with the sensor frame in Waymo's
      x-forward/y-left/z-up convention.  Waymo's origin is the rear axle, so
      ``preprocess`` shifts it by 1.44 m; ours is already the box centre, so the
      translation is ``camera.pos`` verbatim.
    * ``image_sizes [3, 2]`` -- ``(height, width)``, in that order.
    """
    intrinsics = np.zeros((3, 9), dtype=np.float64)
    extrinsics = np.zeros((3, 4, 4), dtype=np.float64)
    sizes = np.zeros((3, 2), dtype=np.int32)
    for index, camera in enumerate(rig_cameras()):
        intrinsics[index, :4] = sensor_intrinsics(camera, width, height)
        ego_to_camera = camera.rotation().numpy().astype(np.float64)
        extrinsics[index] = np.eye(4)
        extrinsics[index, :3, :3] = (SENSOR_TO_CV.T @ ego_to_camera).T
        extrinsics[index, :3, 3] = camera.pos
        sizes[index] = (height, width)
    return intrinsics, extrinsics, sizes
