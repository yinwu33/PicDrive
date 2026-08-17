"""Bench2Drive (CARLA leaderboard 2.0) hosting of the giga policy.

The parent package trains and evaluates the policy on routes this repository
generates.  This subpackage is the other host: Bench2Drive's 220 published
safety-critical routes, scored with leaderboard 2.0 criteria, which is the
number other papers report.  Same policy, different loop owner.

    :mod:`export`   freeze the student + planning head into a TorchScript
                    bundle that pins the scored policy at export time
    :mod:`agent`    the ``AutonomousAgent`` the leaderboard instantiates,
                    running the 10 Hz policy against a 20 Hz tick
    :mod:`video`    render the agent's frame dumps into annotated videos

Nothing here imports the simulator-side stack: the agent loads the frozen
bundle only, so a score always names one specific policy.
"""

from __future__ import annotations
