import importlib
import pufferlib.emulation


def lazy_import(module_path, attr):
    """
    Returns a callable that, when called with any arguments, will
    import the module, retrieve the attribute (usually a class or factory)
    and then call it with the given arguments.
    """
    return lambda *args, **kwargs: getattr(__import__(module_path, fromlist=[attr]), attr)(*args, **kwargs)


def make_foraging(
    width=1080, height=720, num_agents=4096, horizon=512, discretize=True, food_reward=0.1, render_mode="rgb_array"
):
    from .grid import grid

    init_fn = grid.init_foraging
    reward_fn = grid.reward_foraging
    return grid.PufferGrid(
        width,
        height,
        num_agents,
        horizon,
        discretize=discretize,
        food_reward=food_reward,
        init_fn=init_fn,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )


def make_predator_prey(
    width=1080, height=720, num_agents=4096, horizon=512, discretize=True, food_reward=0.1, render_mode="rgb_array"
):
    from .grid import grid

    init_fn = grid.init_predator_prey
    reward_fn = grid.reward_predator_prey
    return grid.PufferGrid(
        width,
        height,
        num_agents,
        horizon,
        discretize=discretize,
        food_reward=food_reward,
        init_fn=init_fn,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )


def make_group(
    width=1080, height=720, num_agents=4096, horizon=512, discretize=True, food_reward=0.1, render_mode="rgb_array"
):
    from .grid import grid

    init_fn = grid.init_group
    reward_fn = grid.reward_group
    return grid.PufferGrid(
        width,
        height,
        num_agents,
        horizon,
        discretize=discretize,
        food_reward=food_reward,
        init_fn=init_fn,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )


def make_puffer(
    width=1080, height=720, num_agents=4096, horizon=512, discretize=True, food_reward=0.1, render_mode="rgb_array"
):
    from .grid import grid

    init_fn = grid.init_puffer
    reward_fn = grid.reward_puffer
    return grid.PufferGrid(
        width,
        height,
        num_agents,
        horizon,
        discretize=discretize,
        food_reward=food_reward,
        init_fn=init_fn,
        reward_fn=reward_fn,
        render_mode=render_mode,
    )


def make_puffergrid(
    render_mode="raylib", vision_range=5, num_envs=4096, num_maps=1000, max_map_size=9, report_interval=128, buf=None
):
    return PufferGrid(render_mode, vision_range, num_envs, num_maps, max_map_size, report_interval, buf)


def make_continuous(discretize=False, buf=None, **kwargs):
    from . import sanity

    env = sanity.Continuous(discretize=discretize)
    if not discretize:
        env = pufferlib.ClipAction(env)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def make_squared(distance_to_target=3, num_targets=1, buf=None, **kwargs):
    from . import sanity

    env = sanity.Squared(distance_to_target=distance_to_target, num_targets=num_targets, **kwargs)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, **kwargs)


def make_bandit(num_actions=10, reward_scale=1, reward_noise=1, buf=None):
    from . import sanity

    env = sanity.Bandit(num_actions=num_actions, reward_scale=reward_scale, reward_noise=reward_noise)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def make_memory(mem_length=2, mem_delay=2, buf=None, **kwargs):
    from . import sanity

    env = sanity.Memory(mem_length=mem_length, mem_delay=mem_delay)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def make_password(password_length=5, buf=None, **kwargs):
    from . import sanity

    env = sanity.Password(password_length=password_length)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def make_performance(delay_mean=0, delay_std=0, bandwidth=1, buf=None, **kwargs):
    from . import sanity

    env = sanity.Performance(delay_mean=delay_mean, delay_std=delay_std, bandwidth=bandwidth)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def make_performance_empiric(count_n=0, count_std=0, bandwidth=1, buf=None, **kwargs):
    from . import sanity

    env = sanity.PerformanceEmpiric(count_n=count_n, count_std=count_std, bandwidth=bandwidth)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def make_stochastic(p=0.7, horizon=100, buf=None, **kwargs):
    from . import sanity

    env = sanity.Stochastic(p=p, horizon=100)
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf)


def make_spaces(buf=None, **kwargs):
    from . import sanity

    env = sanity.Spaces()
    env = pufferlib.EpisodeStats(env)
    return pufferlib.emulation.GymnasiumPufferEnv(env=env, buf=buf, **kwargs)


def make_multiagent(buf=None, **kwargs):
    from . import sanity

    env = sanity.Multiagent()
    env = pufferlib.MultiagentEpisodeStats(env)
    return pufferlib.emulation.PettingZooPufferEnv(env=env, buf=buf)


MAKE_FUNCTIONS = {
    "drive": "Drive",
    # Same simulator as `drive`, configured for perspective observations and
    # wrapped by PerspectiveVecEnv below.
    "drive_cam": "Drive",
    "drive_3cam": "Drive",
    "spaces": make_spaces,
    "multiagent": make_multiagent,
}

# Environments that reuse another environment's module.
MODULE_ALIASES = {"drive_cam": "drive", "drive_3cam": "drive"}


def env_creator(name="squared", *args, **kwargs):
    if "puffer_" not in name:
        raise pufferlib.APIUsageError(f"Invalid environment name: {name}")

    # TODO: Robust sanity / ocean imports
    name = name.replace("puffer_", "")
    module_name = MODULE_ALIASES.get(name, name)
    try:
        module = importlib.import_module(f"pufferlib.ocean.{module_name}.{module_name}")
        return getattr(module, MAKE_FUNCTIONS[name])
    except ModuleNotFoundError:
        return MAKE_FUNCTIONS[name]


def vecenv_wrapper(env_name, vecenv, args):
    """Insert the perspective observation pipeline for camera-driven envs.

    The rasterizer belongs to the observation pipeline rather than to the policy:
    wrapping here is what keeps the privileged scene out of the network's inputs.
    """
    if env_name not in {"puffer_drive_cam", "puffer_drive_3cam"}:
        return vecenv

    from pufferlib.ocean.drive.perspective import PerspectiveVecEnv

    return PerspectiveVecEnv(
        vecenv,
        cameras=args["env"].get("cameras"),
        device=args["train"]["device"],
    )
