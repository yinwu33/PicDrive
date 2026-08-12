"""Perspective observation pipeline for PufferDrive (Pictura reproduction).

`PerspectiveVecEnv` sits between the vectorised simulator and the trainer. It
reads the world-frame RenderState the C environment writes, rasterises it on the
GPU into each ego's camera views, and hands the trainer an observation made of
those pixels plus the ego vector.

This wrapper is the privilege barrier. The scene primitives -- exact poses,
extents and headings of every surrounding agent, and the full road graph -- are
consumed here and never leave. The policy's `forward` is only ever reached by
rendered pixels and the ego state a real vehicle can measure about itself, which
is what makes this perspective self-play rather than a privileged policy wearing
a camera.

Observation layout, packed into one uint8 buffer so a single rollout tensor holds
both parts without paying float32 for the images:

    [ num_cameras * 3 * H * W image bytes | ego_dim float32 ego values ]

`TeddyDriveCam` in pufferlib/teddy/torch.py unpacks it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium
import numpy as np
import torch

import pufferlib
from pufferlib.teddy.drive import binding, raster_cuda
from pufferlib.teddy.drive.raster_ref import DEFAULT_RIG, Camera, rig_from_config, rig_tensor

# Bytes per panel label handed to the viewer, NUL-padded. Long enough for the
# names in the rigs plus a terminator.
_NAME_STRIDE = 16


@dataclass(frozen=True)
class RenderNoise:
    """Gaussian pose jitter applied to the rasterizer's copy of every agent.

    x/y are additive, in metres. z has no dedicated column in the render
    contract -- `RENDER_AGENT_FEATURES` has no agent z position, since the world
    is flat and every box is drawn from ground z=0 up to its `height` -- so z
    noise perturbs that `height` column instead. Heading noise samples an angle
    in degrees and rotates the stored (cos_h, sin_h) unit vector by it, rather
    than adding to cos_h/sin_h independently, which would not stay a unit vector.

    None of this reaches the simulator: it is applied to a tensor already copied
    off the C environment's RenderState buffers (see `PerspectiveVecEnv._render`),
    well after anything that feeds physics, reward or the ego observation.
    """

    enabled: bool = True
    x_mean: float = 0.0
    x_std: float = 0.0
    y_mean: float = 0.0
    y_std: float = 0.0
    z_mean: float = 0.0
    z_std: float = 0.0
    heading_mean_deg: float = 0.0
    heading_std_deg: float = 0.0

    @classmethod
    def from_env_config(cls, cfg: dict) -> "RenderNoise":
        """Build from the `[env]` section's `render_noise_*` keys, all optional."""
        return cls(
            enabled=bool(cfg.get("render_noise_enabled", True)),
            x_mean=float(cfg.get("render_noise_x_mean", 0.0)),
            x_std=float(cfg.get("render_noise_x_std", 0.0)),
            y_mean=float(cfg.get("render_noise_y_mean", 0.0)),
            y_std=float(cfg.get("render_noise_y_std", 0.0)),
            z_mean=float(cfg.get("render_noise_z_mean", 0.0)),
            z_std=float(cfg.get("render_noise_z_std", 0.0)),
            heading_mean_deg=float(cfg.get("render_noise_heading_mean_deg", 0.0)),
            heading_std_deg=float(cfg.get("render_noise_heading_std_deg", 0.0)),
        )

    @property
    def active(self) -> bool:
        """Whether applying this would change anything."""
        return self.enabled and any(
            v != 0.0
            for v in (
                self.x_mean,
                self.x_std,
                self.y_mean,
                self.y_std,
                self.z_mean,
                self.z_std,
                self.heading_mean_deg,
                self.heading_std_deg,
            )
        )


def apply_render_noise(agents: torch.Tensor, noise: RenderNoise) -> None:
    """Jitter `agents` (`[A, 8] = x, y, cos_h, sin_h, length, width, height, type`) in place.

    A no-op axis (mean and std both 0) is skipped rather than sampling a
    zero-variance normal, so a `RenderNoise()` default costs nothing.
    """
    n = agents.shape[0]
    device = agents.device
    if noise.x_mean or noise.x_std:
        agents[:, 0] += torch.randn(n, device=device) * noise.x_std + noise.x_mean
    if noise.y_mean or noise.y_std:
        agents[:, 1] += torch.randn(n, device=device) * noise.y_std + noise.y_mean
    if noise.z_mean or noise.z_std:
        agents[:, 6] += torch.randn(n, device=device) * noise.z_std + noise.z_mean
        # A zero or negative height degenerates the box into nothing the
        # rasterizer can draw a face for.
        agents[:, 6].clamp_(min=0.05)
    if noise.heading_mean_deg or noise.heading_std_deg:
        mean_rad = math.radians(noise.heading_mean_deg)
        std_rad = math.radians(noise.heading_std_deg)
        dtheta = torch.randn(n, device=device) * std_rad + mean_rad
        cos_h, sin_h = agents[:, 2].clone(), agents[:, 3].clone()
        cos_d, sin_d = torch.cos(dtheta), torch.sin(dtheta)
        agents[:, 2] = cos_h * cos_d - sin_h * sin_d
        agents[:, 3] = cos_h * sin_d + sin_h * cos_d


def display_order(cameras: list[Camera]) -> list[int]:
    """Panel order for the viewer: left to right across the rig.

    That is descending mounting yaw, since positive yaw turns left, so the Waymo
    rig comes out front_left, front, front_right. This only orders what the
    viewer blits -- the policy sees the rig in config order either way. A rear
    camera has nowhere natural to go in a left-to-right strip and lands at the
    left end; the rigs in use have none.
    """
    return sorted(range(len(cameras)), key=lambda i: -cameras[i].yaw_deg)


class PerspectiveVecEnv:
    """Wraps a Drive vecenv running in `render_state` mode.

    Implements the subset of the vecenv protocol the trainer uses:
    `async_reset`, `recv`, `send`, `reset`, `step`, `close`, the four space
    attributes, `num_agents`, `agents_per_batch` and `driver_env`.
    """

    def __init__(
        self,
        vecenv,
        cameras: list[Camera] | str | None = None,
        device: str = "cuda",
        render_noise: RenderNoise | None = None,
    ):
        self.vecenv = vecenv
        self.device = torch.device(device)
        self.render_noise = render_noise or RenderNoise()
        if self.device.type != "cuda":
            raise ValueError("PerspectiveVecEnv needs a CUDA device for the rasterizer")

        # The config may arrive as Camera objects, as a preset name, or as the
        # list of dicts the ini parser produces from the JSON rig.
        if isinstance(cameras, list) and cameras and all(isinstance(c, Camera) for c in cameras):
            self.cameras = list(cameras)
        else:
            self.cameras = rig_from_config(cameras) if cameras else list(DEFAULT_RIG)
        heights = {c.height for c in self.cameras}
        widths = {c.width for c in self.cameras}
        if len(heights) != 1 or len(widths) != 1:
            raise ValueError("All cameras in a rig must share a resolution")
        self.height, self.width = heights.pop(), widths.pop()
        self.num_cameras = len(self.cameras)

        self.envs = self._collect_envs(vecenv)
        for env in self.envs:
            if getattr(env, "obs_mode", None) != "render_state":
                raise ValueError("PerspectiveVecEnv requires the Drive env in obs_mode='render_state'")

        self.ego_dim = self.envs[0].ego_features
        self.image_bytes = self.num_cameras * 3 * self.height * self.width
        if self.image_bytes % 4 != 0:
            raise ValueError("Image byte count must be 4-aligned so the ego vector can be bitcast")
        self.obs_bytes = self.image_bytes + self.ego_dim * 4

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(self.obs_bytes,), dtype=np.uint8
        )
        self.single_action_space = vecenv.single_action_space
        self.num_agents = vecenv.num_agents
        self.observation_space = pufferlib.spaces.joint_space(
            self.single_observation_space, self.num_agents
        )
        self.action_space = vecenv.action_space

        self._rig = rig_tensor(self.cameras, device=self.device)
        self._obs = torch.empty((self.num_agents, self.obs_bytes), dtype=torch.uint8, device=self.device)
        self._images = torch.empty(
            (self.num_agents, self.num_cameras, 3, self.height, self.width),
            dtype=torch.uint8,
            device=self.device,
        )
        self._roads_gpu = None
        self._roads_key = None
        self._road_ranges = None
        # Host-side staging for the viewer's camera panels, bound to the C env on
        # first use so raylib can upload it as a texture.
        self._camera_rgb = None
        self._camera_names = None
        self._camera_env_id = None
        self._display_order = display_order(self.cameras)

    @staticmethod
    def _collect_envs(vecenv):
        """Find the Drive instances holding the RenderState buffers.

        The Serial backend exposes them as `.envs`; the PufferEnv backend, used by
        the evaluator, hands back the environment itself.
        """
        envs = getattr(vecenv, "envs", None)
        if envs is not None:
            return list(envs)
        if hasattr(vecenv, "render_state"):
            return [vecenv]
        driver = getattr(vecenv, "driver_env", None)
        if driver is not None and hasattr(driver, "render_state"):
            return [driver]
        raise ValueError(
            "PerspectiveVecEnv needs in-process envs. Use the Serial or PufferEnv "
            "backend; the shared-memory path for Multiprocessing is not built yet."
        )

    # -- properties the trainer reads ---------------------------------------

    @property
    def agents_per_batch(self):
        return self.vecenv.agents_per_batch

    @property
    def driver_env(self):
        return self

    @property
    def emulated(self):
        return False

    @property
    def num_envs(self):
        return sum(env.num_envs for env in self.envs)

    def __getattr__(self, name):
        """Delegate anything unhandled to the underlying Drive environment.

        Callers such as the evaluator reach through `driver_env` for simulator
        facilities -- `render`, `resample_maps`, `get_global_agent_state` -- that
        have nothing to do with observations. Forwarding them keeps this wrapper
        transparent for everything except the observation itself.
        """
        if name.startswith("_") or "envs" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["envs"][0], name)

    # -- scene assembly ------------------------------------------------------

    def _scenes(self):
        """Yield (render_state, num_egos) for every C environment, in agent order."""
        for env in self.envs:
            for i, rs in enumerate(env.render_state):
                num_egos = env.agent_offsets[i + 1] - env.agent_offsets[i]
                yield rs, num_egos

    def _upload_roads(self):
        """Upload road geometry, which only changes when the maps are resampled."""
        # Identity, held by reference rather than by id(): a resample frees the old
        # buffers, and CPython would happily hand their addresses to the new ones,
        # making a stale cache look fresh. Keeping the arrays alive makes `is` exact.
        key = [rs["roads"] for rs, _ in self._scenes()]
        if self._roads_key is not None and len(key) == len(self._roads_key):
            if all(a is b for a, b in zip(key, self._roads_key)):
                return
        chunks, offsets, total = [], [0], 0
        for rs, _ in self._scenes():
            n = int(rs["counts"][1])
            chunks.append(rs["roads"][:n])
            total += n
            offsets.append(total)
        roads = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 6), dtype=np.float32)
        self._roads_gpu = torch.from_numpy(np.ascontiguousarray(roads)).to(self.device)
        self._road_ranges = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        self._roads_key = key

    def _render(self):
        """Rasterise the current scene into `self._obs`, then drop the scene."""
        self._upload_roads()

        agent_chunks, ego_chunks, agent_offsets, scene_ids = [], [], [0], []
        total = 0
        for scene, (rs, num_egos) in enumerate(self._scenes()):
            na = int(rs["counts"][0])
            agent_chunks.append(rs["agents"][:na])
            total += na
            agent_offsets.append(total)
            ego_chunks.append(rs["egos"][:num_egos])
            scene_ids.append(np.full(num_egos, scene, dtype=np.int32))

        agents = np.concatenate(agent_chunks, axis=0)
        egos = np.concatenate(ego_chunks, axis=0)
        scene_ids = np.concatenate(scene_ids, axis=0)

        # Every agent row must get a freshly rendered view. If a map resample ever
        # yields fewer egos than the buffer holds, the tail would silently keep the
        # previous step's pixels, which is the kind of bug that only shows up as
        # mysteriously poor learning.
        if egos.shape[0] != self.num_agents:
            raise RuntimeError(
                f"ego count {egos.shape[0]} does not match the {self.num_agents} agent "
                "rows in the observation buffer"
            )

        agents_t = torch.from_numpy(np.ascontiguousarray(agents)).to(self.device, non_blocking=True)
        egos_t = torch.from_numpy(np.ascontiguousarray(egos)).to(self.device, non_blocking=True)
        ego_scene = torch.from_numpy(scene_ids).to(self.device, non_blocking=True)
        agent_ranges = torch.tensor(agent_offsets, dtype=torch.int32, device=self.device)

        # `agents_t` is already a copy off the C buffers (the `.to` above), so
        # jittering it here reaches only the rasterizer, never the real Entity
        # state `env.step` and reward computed against.
        if self.render_noise.active:
            apply_render_noise(agents_t, self.render_noise)

        raster_cuda.render(
            agents_t,
            self._roads_gpu,
            egos_t,
            cameras=self.cameras,
            rig=self._rig,
            out=self._images,
            ego_scene=ego_scene,
            agent_ranges=agent_ranges,
            road_ranges=self._road_ranges,
        )

        # Pack pixels into the single uint8 observation. Nothing else crosses this
        # line: the scene tensors above go out of scope here, so the policy cannot
        # reach the poses, extents or road graph they carry.
        self._obs[:, : self.image_bytes] = self._images.reshape(self.num_agents, -1)

    def _pack_ego(self, raw_obs):
        """Copy the ego vector, which the C env wrote as the whole observation."""
        ego = torch.as_tensor(raw_obs).to(self.device, non_blocking=True)
        self._obs[:, self.image_bytes :] = ego.contiguous().view(torch.uint8)

    # -- vecenv protocol -----------------------------------------------------

    def async_reset(self, seed=None):
        self.vecenv.async_reset(seed)

    def recv(self):
        obs, rewards, terminals, truncations, infos, agent_ids, masks = self.vecenv.recv()
        self._render()
        self._pack_ego(obs)
        return self._obs, rewards, terminals, truncations, infos, agent_ids, masks

    def send(self, actions):
        self.vecenv.send(actions)

    def reset(self, seed=42):
        self.async_reset(seed)
        obs, _, _, _, infos, _, _ = self.recv()
        return obs, infos

    def step(self, actions, per_env_logs=False):
        # The evaluator asks for per-environment logs, which only Drive.step
        # accepts; the Serial backend and PufferEnv.send do not take it.
        try:
            raw = self.vecenv.step(actions, per_env_logs=per_env_logs)
        except TypeError:
            raw = self.vecenv.step(actions)
        self._render()
        self._pack_ego(raw[0])
        return (self._obs,) + tuple(raw[1:])

    def render(self, *args, **kwargs):
        """Render the simulator, with the selected agent's camera views overlaid.

        The panels are blitted straight from the rasterizer's output, so the
        viewer shows the observation the policy is acting on rather than a
        separately drawn approximation of it.
        """
        env = self.envs[0]
        agent = self._render_agent_index(env, kwargs.get("env_id", 0))
        if self._camera_rgb is None:
            self._camera_rgb = np.zeros(
                (self.num_cameras, self.height, self.width, 3), dtype=np.uint8
            )
            self._camera_names = np.zeros((self.num_cameras, _NAME_STRIDE), dtype=np.uint8)
            for row, cam in enumerate(self._display_order):
                name = self.cameras[cam].name.encode("ascii", "replace")[: _NAME_STRIDE - 1]
                self._camera_names[row, : len(name)] = np.frombuffer(name, dtype=np.uint8)
            binding.env_put(
                env.env_ids[0],
                render_camera_rgb=self._camera_rgb,
                render_camera_names=self._camera_names,
            )
            self._camera_env_id = env.env_ids[0]

        # [cameras, 3, H, W] on device -> [cameras, H, W, 3] on host in panel
        # order, which is the layout raylib uploads.
        views = (
            self._images[agent, self._display_order].permute(0, 2, 3, 1).contiguous().cpu().numpy()
        )
        np.copyto(self._camera_rgb, views)
        return env.render(*args, **kwargs)

    @staticmethod
    def _render_agent_index(env, env_id):
        """Global agent row of the first controlled agent in the rendered env."""
        if 0 <= env_id < len(env.agent_offsets) - 1:
            return int(env.agent_offsets[env_id])
        return 0

    def close(self):
        self.vecenv.close()
