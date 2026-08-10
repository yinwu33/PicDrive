"""Environment registry for the `giga` package.

This is a fork of `pufferlib.ocean` carrying only the driving environment, so that
the Gigaflow-style random initialization can diverge from the dataset-driven env in
`pufferlib/ocean/` without touching it. The two are independent: `ocean` keeps
serving WOSAC and human-replay evaluation, which need logged trajectories.

Env names are `puffer_giga` (privileged vector observations), `puffer_giga_cam`
(a single front camera) and `puffer_giga_3cam` (Waymo's three front-facing
cameras). All three are the same C simulator in different observation modes,
exactly as in `ocean`, where the counterparts are `puffer_drive`,
`puffer_drive_cam` and `puffer_drive_3cam`.
"""

import importlib

import pufferlib

MAKE_FUNCTIONS = {
    "giga": "Drive",
    # Same simulator, configured for perspective observations and wrapped by
    # PerspectiveVecEnv below. These two differ only in the camera rig named by
    # their config, so they share this entry point.
    "giga_cam": "Drive",
    "giga_3cam": "Drive",
}

# Environments that reuse another environment's module.
MODULE_ALIASES = {name: "drive" for name in MAKE_FUNCTIONS}

# Envs whose observations are rendered rather than privileged. Matched exactly:
# every name here starts with "giga", so a substring test would wrap the
# vector-observation env too.
CAMERA_ENVS = {"puffer_giga_cam", "puffer_giga_3cam"}


def env_creator(name="giga", *args, **kwargs):
    if "puffer_" not in name:
        raise pufferlib.APIUsageError(f"Invalid environment name: {name}")

    name = name.replace("puffer_", "")
    if name not in MAKE_FUNCTIONS:
        raise pufferlib.APIUsageError(
            f"Unknown giga environment {name!r}. Valid names: {sorted(MAKE_FUNCTIONS)}"
        )

    module_name = MODULE_ALIASES.get(name, name)
    # Deliberately not catching ModuleNotFoundError: a missing compiled binding is a
    # real build failure, and swallowing it here surfaces much later as a confusing
    # "str is not callable".
    module = importlib.import_module(f"pufferlib.giga.{module_name}.{module_name}")
    return getattr(module, MAKE_FUNCTIONS[name])


def vecenv_wrapper(env_name, vecenv, args):
    """Insert the perspective observation pipeline for camera-driven envs.

    The rasterizer belongs to the observation pipeline rather than to the policy:
    wrapping here is what keeps the privileged scene out of the network's inputs.
    """
    if env_name not in CAMERA_ENVS:
        return vecenv

    from pufferlib.giga.drive.perspective import PerspectiveVecEnv

    return PerspectiveVecEnv(
        vecenv,
        cameras=args["env"].get("cameras"),
        device=args["train"]["device"],
    )
