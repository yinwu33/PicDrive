from torch import nn
import torch
import torch.nn.functional as F

import pufferlib
import pufferlib.models
import pufferlib.pytorch

from pufferlib.models import Default as Policy  # noqa: F401
from pufferlib.models import Convolutional as Conv  # noqa: F401


Recurrent = pufferlib.models.LSTMWrapper


class Drive(nn.Module):
    def __init__(self, env, input_size=128, hidden_size=128, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_size = env.single_observation_space.shape[0]
        self.max_partner_objects = env.max_partner_objects
        self.partner_features = env.partner_features
        self.max_road_objects = env.max_road_objects
        self.road_features = env.road_features
        self.road_features_after_onehot = env.road_features + 6  # 6 is the number of one-hot encoded categories
        self.ego_dim = env.ego_features

        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_dim, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.road_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.road_features_after_onehot, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.partner_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.partner_features, input_size)),
            nn.LayerNorm(input_size),
            # nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, input_size)),
        )

        self.shared_embedding = nn.Sequential(
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Linear(3 * input_size, hidden_size)),
        )
        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)

        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        logits, values = self.decode_actions(hidden)
        return logits, values

    def encode_observations(self, observations, state=None):
        ego_dim = self.ego_dim
        partner_dim = self.max_partner_objects * self.partner_features
        road_dim = self.max_road_objects * self.road_features
        ego_obs = observations[:, :ego_dim]
        partner_obs = observations[:, ego_dim : ego_dim + partner_dim]
        road_obs = observations[:, ego_dim + partner_dim : ego_dim + partner_dim + road_dim]

        partner_objects = partner_obs.view(-1, self.max_partner_objects, self.partner_features)

        road_objects = road_obs.view(-1, self.max_road_objects, self.road_features)
        road_continuous = road_objects[:, :, : self.road_features - 1]
        road_categorical = road_objects[:, :, self.road_features - 1]
        road_onehot = F.one_hot(road_categorical.long(), num_classes=7)  # Shape: [batch, ROAD_MAX_OBJECTS, 7]
        road_objects = torch.cat([road_continuous, road_onehot], dim=2)
        ego_features = self.ego_encoder(ego_obs)
        partner_features, _ = self.partner_encoder(partner_objects).max(dim=1)
        road_features, _ = self.road_encoder(road_objects).max(dim=1)

        concat_features = torch.cat([ego_features, road_features, partner_features], dim=1)

        # Pass through shared embedding
        embedding = F.relu(self.shared_embedding(concat_features))
        # embedding = self.shared_embedding(concat_features)
        return embedding

    def decode_actions(self, flat_hidden):
        if self.is_continuous:
            parameters = self.actor(flat_hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(flat_hidden)
            action = torch.split(action, self.atn_dim, dim=1)

        value = self.value_fn(flat_hidden)

        return action, value


class DriveCam(nn.Module):
    """Perspective driving policy (Alberti, Pictura Sec. 3.3).

    The observation is the packed buffer `PerspectiveVecEnv` produces: camera
    pixels followed by the ego vector. No privileged scene values reach this
    module -- surrounding traffic and road geometry are only available as pixels,
    which is the whole point of perspective self-play.

    Following the paper, a small convolutional stack trained from scratch encodes
    each camera view with shared weights, the per-camera features are projected
    and concatenated into a scene embedding, and that is concatenated with the
    encoded ego vector before a shared actor-critic trunk.

    The paper pools each camera's feature map with 16 learned query tokens that
    cross-attend over it. That exists to map cameras of differing resolution into
    a latent of common width across a four-camera rig; with a single camera the
    motivation does not apply, so this uses a projection of the flattened map and
    leaves token pooling as an ablation for when the rig grows.
    """

    def __init__(
        self,
        env,
        input_size=128,
        hidden_size=512,
        cnn_channels=128,
        scene_dim=256,
        ego_dim=64,
        backbone_layers=4,
        **kwargs,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_cameras = env.num_cameras
        self.cam_height = env.height
        self.cam_width = env.width
        self.image_bytes = env.image_bytes
        self.ego_features = env.ego_dim

        # Five convolutions to 128 channels, shared across cameras, trained from
        # scratch (paper Tab. 4).
        self.cnn = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Conv2d(3, 32, 4, stride=2, padding=1)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(32, 64, 4, stride=2, padding=1)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(64, cnn_channels, 3, stride=2, padding=1)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=2, padding=1)),
            nn.GELU(),
            pufferlib.pytorch.layer_init(nn.Conv2d(cnn_channels, cnn_channels, 3, stride=1, padding=1)),
            nn.GELU(),
        )
        with torch.no_grad():
            probe = torch.zeros(1, 3, self.cam_height, self.cam_width)
            self.cnn_out = self.cnn(probe).flatten(1).shape[1]

        self.cam_proj = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.cnn_out, input_size)),
            nn.LayerNorm(input_size),
            nn.GELU(),
        )
        self.scene_encoder = pufferlib.pytorch.layer_init(
            nn.Linear(input_size * self.num_cameras, scene_dim)
        )
        self.ego_encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(self.ego_features, ego_dim)),
            nn.LayerNorm(ego_dim),
            pufferlib.pytorch.layer_init(nn.Linear(ego_dim, ego_dim)),
        )

        layers = []
        width = scene_dim + ego_dim
        for _ in range(backbone_layers):
            layers += [pufferlib.pytorch.layer_init(nn.Linear(width, hidden_size)), nn.GELU()]
            width = hidden_size
        self.backbone = nn.Sequential(*layers)

        self.is_continuous = isinstance(env.single_action_space, pufferlib.spaces.Box)
        if self.is_continuous:
            self.atn_dim = (env.single_action_space.shape[0],) * 2
        else:
            self.atn_dim = env.single_action_space.nvec.tolist()

        self.actor = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, sum(self.atn_dim)), std=0.01)
        self.value_fn = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, 1), std=1)

    def unpack(self, observations):
        """Split the packed uint8 observation into images and the ego vector."""
        obs = observations if observations.dtype == torch.uint8 else observations.to(torch.uint8)
        obs = obs.contiguous()
        batch = obs.shape[0]
        # reshape, not view: the image slice is a strided window into a row that
        # also carries the ego bytes, so folding the camera axis into the batch
        # spans a discontinuity for any rig with more than one camera.
        images = obs[:, : self.image_bytes].reshape(
            batch * self.num_cameras, 3, self.cam_height, self.cam_width
        )
        ego = obs[:, self.image_bytes :].contiguous().view(torch.float32)
        return images, ego

    def encode_observations(self, observations, state=None):
        images, ego = self.unpack(observations)
        batch = ego.shape[0]

        features = self.cnn(images.float() / 255.0).flatten(1)
        features = self.cam_proj(features).view(batch, -1)
        scene = self.scene_encoder(features)

        hidden = torch.cat([scene, self.ego_encoder(ego)], dim=1)
        return self.backbone(hidden)

    def decode_actions(self, hidden):
        if self.is_continuous:
            parameters = self.actor(hidden)
            loc, scale = torch.split(parameters, self.atn_dim, dim=1)
            std = torch.nn.functional.softplus(scale) + 1e-4
            action = torch.distributions.Normal(loc, std)
        else:
            action = self.actor(hidden)
            action = torch.split(action, self.atn_dim, dim=1)
        return action, self.value_fn(hidden)

    def forward(self, observations, state=None):
        hidden = self.encode_observations(observations, state=state)
        return self.decode_actions(hidden)

    def forward_train(self, x, state=None):
        return self.forward(x, state)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)
