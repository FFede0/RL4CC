"""Per-agent Q-networks and VDN/QMIX mixers."""
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from gymnasium.spaces import Discrete
from gymnasium.spaces.utils import flatdim
import numpy as np

torch, nn = try_import_torch()
F = torch.nn.functional if torch is not None else None


def _activation_module(name):
  if name is None:
    return nn.ReLU()
  n = str(name).lower()
  if n == "elu":
    return nn.ELU()
  if n == "tanh":
    return nn.Tanh()
  return nn.ReLU()


class AgentQNet(nn.Module):
  def __init__(self, obs_dim, n_actions, hidden=(256, 128), activation="relu"):
    super().__init__()
    layers = []
    prev = int(obs_dim)
    act = _activation_module(activation)
    for h in hidden:
      layers += [nn.Linear(prev, int(h)), act]
      prev = int(h)
    layers.append(nn.Linear(prev, int(n_actions)))
    self.net = nn.Sequential(*layers)

  def forward(self, obs):
    return self.net(obs)


class VDNMixer(nn.Module):
  def forward(self, agent_q, _state):
    return agent_q.sum(dim=-1)


class QMixer(nn.Module):
  def __init__(self, n_agents, state_dim, embed_dim=32):
    super().__init__()
    self.n_agents = int(n_agents)
    self.embed_dim = int(embed_dim)
    self.hyper_w1 = nn.Linear(int(state_dim), self.embed_dim * self.n_agents)
    self.hyper_w2 = nn.Linear(int(state_dim), self.embed_dim)
    self.hyper_b1 = nn.Linear(int(state_dim), self.embed_dim)
    self.val_v = nn.Linear(int(state_dim), 1)

  def forward(self, agent_q, state):
    b = agent_q.shape[0]
    w1 = torch.abs(self.hyper_w1(state)).view(b, self.n_agents, self.embed_dim)
    b1 = self.hyper_b1(state).view(b, 1, self.embed_dim)
    hidden = F.elu(torch.bmm(agent_q.unsqueeze(1), w1) + b1)
    w2 = torch.abs(self.hyper_w2(state)).view(b, self.embed_dim, 1)
    b2 = self.val_v(state).view(b, 1, 1)
    return (torch.bmm(hidden, w2) + b2).reshape(b)


def build_mixer(name, n_agents, state_dim, embed_dim=32):
  name = str(name).lower()
  if name == "vdn":
    return VDNMixer()
  if name == "qmix":
    return QMixer(n_agents, state_dim, embed_dim)
  raise ValueError("Unknown mixer '%s'" % name)


def flatten_obs(obs, obs_space):
  if isinstance(obs, dict):
    if hasattr(obs_space, "spaces"):
      keys = list(obs_space.spaces.keys())
    else:
      keys = sorted(obs.keys())
    parts = [np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in keys]
    return np.concatenate(parts, axis=0).astype(np.float32)
  return np.asarray(obs, dtype=np.float32).reshape(-1)


class ValueDecompTorchModel(TorchModelV2, nn.Module):
  """Custom model registered as 'value_decomp_q'."""

  def __init__(
      self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
    TorchModelV2.__init__(
      self, obs_space, action_space, num_outputs, model_config, name
    )
    nn.Module.__init__(self)

    custom = model_config.get("custom_model_config", {}) or {}
    hidden = tuple(
      custom.get("fcnet_hiddens", model_config.get("fcnet_hiddens", [256, 128]))
    )
    activation = custom.get(
      "fcnet_activation", model_config.get("fcnet_activation", "relu")
    )
    self._torch_fc = TorchFC(
      obs_space, action_space, num_outputs, model_config, name + "_fc"
    )
    if isinstance(action_space, Discrete):
      n_actions = int(action_space.n)
    else:
      n_actions = int(num_outputs)
    try:
      obs_dim = int(flatdim(obs_space))
    except Exception:
      obs_dim = int(np.prod(obs_space.shape))
    self.q_net = AgentQNet(obs_dim, n_actions, hidden, activation)
    self._features = None

  @override(TorchModelV2)
  def forward(self, input_dict, state, seq_lens):
    q = self.q_net(input_dict["obs_flat"])
    self._features = q
    return q, state

  @override(TorchModelV2)
  def value_function(self):
    assert self._features is not None
    return torch.zeros(self._features.shape[0], device=self._features.device)
