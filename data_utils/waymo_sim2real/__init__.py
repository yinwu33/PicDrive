"""Waymo Perception preprocessing for PufferDrive sim-to-real training.

The package deliberately separates the raw-data stages from every downstream
consumer:

* :mod:`preprocess` and :mod:`extract_ego_state` are the only modules that read
  Waymo TFRecord files.
* :mod:`extract_teacher_features` reads compact processed ``.npz`` samples.
* :mod:`visualize` reads processed samples and extracted teacher artifacts.
* :mod:`train_distillation` reads only derived artifacts plus the pinned teacher
  checkpoint, whose planning head it holds frozen.
"""

from .processed import CAMERA_NAMES, EGO_OBS_DIM, PROCESSED_SCHEMA_VERSION

__all__ = ["CAMERA_NAMES", "EGO_OBS_DIM", "PROCESSED_SCHEMA_VERSION"]
