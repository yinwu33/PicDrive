"""Interactive closed-loop rollout for watching the distilled policy drive.

This is the visualization host: it drives the ego in CARLA on routes generated
from the parent package's road index, and puts what the policy sees next to what
it does.  ``--camera-preview`` shows CARLA RGB above the matching Puffer teacher
raster, ``--spectator`` keeps a chase camera on the ego, and ``--shadow`` runs
the non-controlling perception branch on the same trajectory so student and
teacher disagreement is readable frame by frame.

Distinct from :mod:`data_utils.sim2real.carla.b2d`, which is the scoring host:
Bench2Drive's published routes under leaderboard 2.0 criteria, no display.

    :mod:`closed_loop`  the rollout, its preview window and its episode log
"""

from __future__ import annotations
