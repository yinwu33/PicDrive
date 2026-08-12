"""Environment registry for the `teddy` package.

`teddy` is the third in-tree driving env, and it is a deliberate crossbreed of the
other two. It is fully self-contained: nothing here imports from `pufferlib.ocean`
or `pufferlib.giga`, and the C sources are copies rather than shared headers, so the
three can be changed independently.

  * The scene comes from `giga`. No logged trajectory supplies anything: agents are
    drawn from the WOMD state distribution, rejection-sampled onto the map's lane
    graph, given a route of up to four waypoints walked over that graph, and
    recycled into a fresh pose and route whenever they finish it.

  * The reward comes from `ocean`. Four fixed terms -- vehicle collision, off-road,
    goal, and a small jerk penalty under classic dynamics -- weighted by config
    values that every agent shares.

That combination is the reason the package exists. `giga` pairs the random scene
with Gigaflow's nine-term reward, weighted per agent by a randomized conditioning
vector, and it has been hard to tell whether a bad policy is answering the scene or
the reward. `teddy` holds the scene fixed and takes the reward complexity away, so
there is only one variable left. Because the reward weights no longer vary per
agent, there is no conditioning at all here: the ego observation is ocean's plain
8 (classic) / 11 (jerk) values, and the dynamics are ocean's, unrandomized.

Env names are `puffer_teddy` (privileged vector observations), `puffer_teddy_cam`
(a single front camera), `puffer_teddy_3cam` (Waymo's three front-facing cameras)
and `puffer_teddy_debug` (the single-car, single-map overfitting rig). All four are
the same C simulator in different observation modes, exactly as in `ocean` and
`giga`.
"""

import importlib

import pufferlib

MAKE_FUNCTIONS = {
    "teddy": "Drive",
    # Same simulator and same observation as `teddy`, pinned by its config to one car
    # on one map. Registered rather than reachable by CLI overrides because the
    # single-scene setup spans [vec], [env] and [train] together, and a half-applied
    # version of it silently trains something else.
    "teddy_debug": "Drive",
    # Same simulator, configured for perspective observations and wrapped by
    # PerspectiveVecEnv below. These two differ only in the camera rig named by
    # their config, so they share this entry point.
    "teddy_cam": "Drive",
    "teddy_3cam": "Drive",
}

# Environments that reuse another environment's module.
MODULE_ALIASES = {name: "drive" for name in MAKE_FUNCTIONS}

# Envs whose observations are rendered rather than privileged. Matched exactly:
# every name here starts with "teddy", so a substring test would wrap the
# vector-observation env too.
CAMERA_ENVS = {"puffer_teddy_cam", "puffer_teddy_3cam"}


def env_creator(name="teddy", *args, **kwargs):
    if "puffer_" not in name:
        raise pufferlib.APIUsageError(f"Invalid environment name: {name}")

    name = name.replace("puffer_", "")
    if name not in MAKE_FUNCTIONS:
        raise pufferlib.APIUsageError(
            f"Unknown teddy environment {name!r}. Valid names: {sorted(MAKE_FUNCTIONS)}"
        )

    module_name = MODULE_ALIASES.get(name, name)
    # Deliberately not catching ModuleNotFoundError: a missing compiled binding is a
    # real build failure, and swallowing it here surfaces much later as a confusing
    # "str is not callable".
    module = importlib.import_module(f"pufferlib.teddy.{module_name}.{module_name}")
    return getattr(module, MAKE_FUNCTIONS[name])


def vecenv_wrapper(env_name, vecenv, args):
    """Insert the perspective observation pipeline for camera-driven envs.

    The rasterizer belongs to the observation pipeline rather than to the policy:
    wrapping here is what keeps the privileged scene out of the network's inputs.
    """
    if env_name not in CAMERA_ENVS:
        return vecenv

    from pufferlib.teddy.drive.perspective import PerspectiveVecEnv, RenderNoise

    return PerspectiveVecEnv(
        vecenv,
        cameras=args["env"].get("cameras"),
        device=args["train"]["device"],
        render_noise=RenderNoise.from_env_config(args["env"]),
    )
