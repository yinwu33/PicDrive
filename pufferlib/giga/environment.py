"""Environment registry for the `giga` package.

This is a fork of `pufferlib.ocean` carrying only the driving environment, so that
the Gigaflow-style random initialization can diverge from the dataset-driven env in
`pufferlib/ocean/` without touching it. The two are independent: `ocean` keeps
serving WOSAC and human-replay evaluation, which need logged trajectories.

Env names are `puffer_giga_drive` (privileged vector observations) and
`puffer_giga_drive_cam` (perspective camera observations). Both are the same C
simulator in different observation modes, exactly as in `ocean`.
"""

import importlib

import pufferlib

MAKE_FUNCTIONS = {
    "giga_drive": "Drive",
    # Same simulator, configured for perspective observations and wrapped by
    # PerspectiveVecEnv below.
    "giga_drive_cam": "Drive",
}

# Environments that reuse another environment's module.
MODULE_ALIASES = {"giga_drive": "drive", "giga_drive_cam": "drive"}


def env_creator(name="giga_drive", *args, **kwargs):
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
    if "giga_drive_cam" not in env_name:
        return vecenv

    from pufferlib.giga.drive.perspective import PerspectiveVecEnv

    return PerspectiveVecEnv(
        vecenv,
        cameras=args["env"].get("cameras"),
        device=args["train"]["device"],
    )
