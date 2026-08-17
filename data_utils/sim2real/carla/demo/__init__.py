"""Interactive closed-loop rollout for watching the distilled policy drive.

Puts what the policy sees next to what it does: ``--camera-preview`` shows CARLA
RGB above the matching Puffer teacher raster, ``--shadow`` runs the
non-controlling perception branch on the same trajectory so student/teacher
disagreement is readable frame by frame.  Distinct from
:mod:`data_utils.sim2real.carla.b2d`, which is the scoring host.
"""

from __future__ import annotations
