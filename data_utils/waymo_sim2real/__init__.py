"""Waymo Perception preprocessing for PufferDrive sim-to-real training.

The package deliberately separates the raw-data stage from every downstream
consumer:

* :mod:`preprocess` is the only module that reads Waymo TFRecord files.
* :mod:`extract_teacher_features` reads compact processed ``.npz`` samples.
* :mod:`visualize` reads processed samples and extracted teacher artifacts.
"""

from .processed import CAMERA_NAMES, PROCESSED_SCHEMA_VERSION

__all__ = ["CAMERA_NAMES", "PROCESSED_SCHEMA_VERSION"]
