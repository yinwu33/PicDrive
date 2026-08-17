"""Bench2Drive (CARLA leaderboard 2.0) hosting of the giga policy.

The parent package evaluates on routes this repository generates; this
subpackage is the other host: Bench2Drive's 220 published safety-critical
routes, scored with leaderboard 2.0 criteria, which is the number other papers
report.  ``export`` freezes the policy into a TorchScript bundle, ``agent`` is
the ``AutonomousAgent`` the leaderboard instantiates, ``video`` renders its
frame dumps.  Nothing here imports the simulator-side stack, so a score always
names one specific frozen policy.
"""

from __future__ import annotations
