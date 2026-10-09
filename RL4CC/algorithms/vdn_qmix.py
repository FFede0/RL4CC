"""
VDN / QMIX for Ray 2.20 (stock QMIX is not available).

Registered like MAPPO. Needs a multi-agent env with Discrete actions and a
shared team reward. For QMIX, prefer infos['__common__']['global_state'] or
env.build_global_state().
"""
import copy
import random

import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from gymnasium.spaces import Discrete
from gymnasium.spaces.utils import flatdim

from RL4CC.models.value_decomp_model import AgentQNet, build_mixer, flatten_obs

torch, nn = try_import_torch()
F = torch.nn.functional if torch is not None else None


def _unwrap_env(env):
  seen = set()
  cur = env
  best = env
  while cur is not None and id(cur) not in seen:
    seen.add(id(cur))
    if hasattr(cur, "agents") and hasattr(cur, "step") and hasattr(cur, "reset"):
      best = cur
    nxt = getattr(cur, "env", None)
    if nxt is None or nxt is cur:
      unwrapped = getattr(cur, "unwrapped", None)
      if unwrapped is not None and unwrapped is not cur:
        cur = unwrapped
        continue
      break
    cur = nxt
  return best


class _ReplayBuffer:
  def __init__(self, capacity):
    self.capacity = int(capacity)
    self.data = []
    self.pos = 0

  def push(self, transition):
    if len(self.data) < self.capacity:
      self.data.append(transition)
    else:
      self.data[self.pos] = transition
    self.pos = (self.pos + 1) % self.capacity

  def sample(self, batch_size):
    batch = random.sample(self.data, int(batch_size))
    return map(np.stack, zip(*batch))

  def __len__(self):
    return len(self.data)


class ValueDecompPolicy(Policy):
  """Policy shell for RLlib; training is done in VDNQMIX."""

  def __init__(self, observation_space, action_space, config):
    Policy.__init__(self, observation_space, action_space, config)
    self.framework = "torch"
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_cfg = config.get("model", {}) or {}
    custom = model_cfg.get("custom_model_config", {}) or {}
    hidden = tuple(
      custom.get("fcnet_hiddens", model_cfg.get("fcnet_hiddens", [256, 128]))
    )
    activation = custom.get(
      "fcnet_activation", model_cfg.get("fcnet_activation", "relu")
    )
    try:
      obs_dim = int(flatdim(observation_space))
    except Exception:
      obs_dim = int(np.prod(observation_space.shape))
    if not isinstance(action_space, Discrete):
      raise ValueError("VDN/QMIX policies require Discrete action spaces")
    n_actions = int(action_space.n)

    self.q_net = AgentQNet(obs_dim, n_actions, hidden, activation).to(self.device)
    self._obs_space = observation_space
    self._n_actions = n_actions
    self._epsilon = float(config.get("epsilon_start", 1.0))

  def set_epsilon(self, epsilon):
    self._epsilon = float(epsilon)

  def compute_actions(
      self,
      obs_batch,
      state_batches=None,
      prev_action_batch=None,
      prev_reward_batch=None,
      info_batch=None,
      episodes=None,
      explore=None,
      timestep=None,
      **kwargs,
    ):
    explore = self.config.get("explore", True) if explore is None else explore
    obs_batch = np.asarray(obs_batch)
    if obs_batch.dtype == object or isinstance(obs_batch[0], dict):
      flat = np.stack(
        [flatten_obs(o, self._obs_space) for o in obs_batch], axis=0
      )
    else:
      flat = obs_batch.reshape(obs_batch.shape[0], -1).astype(np.float32)

    actions = np.zeros(flat.shape[0], dtype=np.int64)
    with torch.no_grad():
      q = self.q_net(
        torch.as_tensor(flat, dtype=torch.float32, device=self.device)
      )
      greedy = q.argmax(dim=-1).cpu().numpy()
    for i in range(flat.shape[0]):
      if explore and random.random() < self._epsilon:
        actions[i] = random.randrange(self._n_actions)
      else:
        actions[i] = int(greedy[i])
    return actions, [], {}

  def learn_on_batch(self, samples):
    return {}

  def get_weights(self):
    return {k: v.cpu().numpy() for k, v in self.q_net.state_dict().items()}

  def set_weights(self, weights):
    self.q_net.load_state_dict(
      {k: torch.as_tensor(v, device=self.device) for k, v in weights.items()}
    )

  def get_state(self):
    return {"weights": self.get_weights(), "epsilon": self._epsilon}

  def set_state(self, state):
    self.set_weights(state["weights"])
    self._epsilon = float(state.get("epsilon", self._epsilon))


class VDNQMIXConfig(AlgorithmConfig):
  def __init__(self, algo_class=None):
    super().__init__(algo_class=algo_class or VDNQMIX)
    self.framework_str = "torch"
    for flag in (
      "_enable_rl_module_and_learner",
      "_enable_env_runner_and_connector_v2",
      "enable_rl_module_and_learner",
      "enable_env_runner_and_connector_v2",
    ):
      try:
        setattr(self, flag, False)
      except Exception:
        pass

    self.mixer = "vdn"
    self.lr = 5e-4
    self.gamma = 0.99
    self.train_batch_size = 64
    self.replay_buffer_capacity = 100000
    self.target_network_update_freq = 500
    self.mixing_embed_dim = 32
    self.grad_clip = 10.0
    self.episodes_per_iter = 2
    self.gradient_steps_per_iter = 30
    self.epsilon_start = 1.0
    self.epsilon_end = 0.05
    self.epsilon_decay_iters = 10000
    self.epsilon_decay_power = 1.0
    self.state_dim = None

    self.model["fcnet_hiddens"] = [256, 128]
    self.model["fcnet_activation"] = "relu"
    self.model["custom_model"] = "value_decomp_q"

    self.num_rollout_workers = 0
    self.num_envs_per_worker = 1
    self.rollout_fragment_length = 1
    self.batch_mode = "complete_episodes"
    self.simple_optimizer = True

  def training(
      self,
      *,
      mixer=None,
      mixing_embed_dim=None,
      replay_buffer_capacity=None,
      target_network_update_freq=None,
      episodes_per_iter=None,
      gradient_steps_per_iter=None,
      epsilon_start=None,
      epsilon_end=None,
      epsilon_decay_iters=None,
      epsilon_decay_power=None,
      state_dim=None,
      grad_clip=None,
      **kwargs,
    ):
    if mixer is not None:
      self.mixer = mixer
    if mixing_embed_dim is not None:
      self.mixing_embed_dim = mixing_embed_dim
    if replay_buffer_capacity is not None:
      self.replay_buffer_capacity = replay_buffer_capacity
    if target_network_update_freq is not None:
      self.target_network_update_freq = target_network_update_freq
    if episodes_per_iter is not None:
      self.episodes_per_iter = episodes_per_iter
    if gradient_steps_per_iter is not None:
      self.gradient_steps_per_iter = gradient_steps_per_iter
    if epsilon_start is not None:
      self.epsilon_start = epsilon_start
    if epsilon_end is not None:
      self.epsilon_end = epsilon_end
    if epsilon_decay_iters is not None:
      self.epsilon_decay_iters = epsilon_decay_iters
    if epsilon_decay_power is not None:
      self.epsilon_decay_power = epsilon_decay_power
    if state_dim is not None:
      self.state_dim = state_dim
    if grad_clip is not None:
      self.grad_clip = grad_clip
    return super().training(**kwargs)


class VDNQMIX(Algorithm):
  @classmethod
  def get_default_config(cls):
    return VDNQMIXConfig()

  @classmethod
  def get_default_policy_class(cls, config):
    return ValueDecompPolicy

  @override(Algorithm)
  def setup(self, config):
    super().setup(config)
    self._vdn_initialized = False
    self._train_iteration = 0
    self._device = torch.device(
      "cuda" if torch.cuda.is_available() else "cpu"
    )
    self._init_value_decomp_modules()

  def _agent_ids(self):
    env = _unwrap_env(self.workers.local_worker().env)
    if hasattr(env, "agents"):
      return list(env.agents)
    return [
      pid for pid in self.workers.local_worker().policy_map.keys()
      if not str(pid).startswith("_")
    ]

  def _init_value_decomp_modules(self):
    if self._vdn_initialized:
      return
    cfg = self.config
    env = _unwrap_env(self.workers.local_worker().env)
    agents = self._agent_ids()
    self._agents = agents
    self._n_agents = len(agents)

    if hasattr(env, "single_agent_observation_space"):
      obs_space = env.single_agent_observation_space
    else:
      obs_space = env.observation_space[agents[0]]
    if hasattr(env, "single_agent_action_space"):
      act_space = env.single_agent_action_space
    else:
      act_space = env.action_space[agents[0]]

    try:
      obs_dim = int(flatdim(obs_space))
    except Exception:
      obs_dim = int(np.prod(obs_space.shape))
    if not isinstance(act_space, Discrete):
      raise ValueError("VDN/QMIX require Discrete per-agent action spaces")
    n_actions = int(act_space.n)

    hidden = tuple(cfg.model.get("fcnet_hiddens", [256, 128]))
    activation = cfg.model.get("fcnet_activation", "relu")
    self._obs_space = obs_space
    self._obs_dim = obs_dim
    self._n_actions = n_actions

    self.q_nets = nn.ModuleList([
      AgentQNet(obs_dim, n_actions, hidden, activation).to(self._device)
      for _ in range(self._n_agents)
    ])
    self.target_q_nets = copy.deepcopy(self.q_nets)

    state_dim = cfg.state_dim
    if state_dim is None:
      if hasattr(env, "_global_state_dim"):
        state_dim = int(env._global_state_dim())
      elif hasattr(env, "build_global_state"):
        state_dim = int(np.asarray(env.build_global_state()).reshape(-1).shape[0])
      else:
        state_dim = obs_dim * self._n_agents
    self._state_dim = int(state_dim)

    self.mixer = build_mixer(
      str(cfg.mixer).lower(), self._n_agents, self._state_dim,
      int(cfg.mixing_embed_dim)
    ).to(self._device)
    self.target_mixer = copy.deepcopy(self.mixer)

    params = list(self.q_nets.parameters()) + list(self.mixer.parameters())
    self._optimizer = torch.optim.Adam(params, lr=float(cfg.lr))
    self._buffer = _ReplayBuffer(int(cfg.replay_buffer_capacity))
    self._vdn_initialized = True
    self._sync_weights_to_policies()

  def _sync_weights_to_policies(self):
    local = self.workers.local_worker()
    for i, agent_id in enumerate(self._agents):
      policy = None
      if agent_id in local.policy_map:
        policy = local.policy_map[agent_id]
      elif len(local.policy_map) == 1:
        policy = next(iter(local.policy_map.values()))
      elif "shared_policy" in local.policy_map:
        if i != 0:
          continue
        policy = local.policy_map["shared_policy"]
      if policy is not None and hasattr(policy, "set_weights"):
        policy.set_weights({
          k: v.detach().cpu().numpy()
          for k, v in self.q_nets[i].state_dict().items()
        })

  def _epsilon(self):
    cfg = self.config
    it = self._train_iteration
    decay = int(cfg.epsilon_decay_iters)
    if it >= decay:
      return float(cfg.epsilon_end)
    t = it / max(decay, 1)
    frac = t ** max(float(cfg.epsilon_decay_power), 1e-6)
    return float(cfg.epsilon_start) + frac * (
      float(cfg.epsilon_end) - float(cfg.epsilon_start)
    )

  def _global_state(self, env, obs_dict, infos):
    if infos and isinstance(infos, dict):
      common = infos.get("__common__") or {}
      if "global_state" in common:
        return np.asarray(common["global_state"], dtype=np.float32).reshape(-1)
      for aid in self._agents:
        if aid in infos and isinstance(infos[aid], dict):
          if "global_state" in infos[aid]:
            return np.asarray(
              infos[aid]["global_state"], dtype=np.float32
            ).reshape(-1)
    if hasattr(env, "build_global_state"):
      return np.asarray(env.build_global_state(), dtype=np.float32).reshape(-1)
    parts = [flatten_obs(obs_dict[a], self._obs_space) for a in self._agents]
    flat = np.concatenate(parts, axis=0).astype(np.float32)
    if flat.shape[0] < self._state_dim:
      flat = np.pad(flat, (0, self._state_dim - flat.shape[0]))
    return flat[: self._state_dim]

  def _fit_state(self, s):
    s = np.asarray(s, dtype=np.float32).reshape(-1)
    if s.shape[0] < self._state_dim:
      s = np.pad(s, (0, self._state_dim - s.shape[0]))
    return s[: self._state_dim]

  def _select_actions(self, obs_dict, epsilon):
    actions = {}
    with torch.no_grad():
      for i, aid in enumerate(self._agents):
        o = flatten_obs(obs_dict[aid], self._obs_space)
        if random.random() < epsilon:
          actions[aid] = random.randrange(self._n_actions)
        else:
          q = self.q_nets[i](
            torch.as_tensor(
              o, dtype=torch.float32, device=self._device
            ).unsqueeze(0)
          )
          actions[aid] = int(q.argmax(dim=-1).item())
    return actions

  def _run_episode(self, env, epsilon):
    reset_out = env.reset()
    if isinstance(reset_out, tuple):
      obs_dict, infos = reset_out
    else:
      obs_dict, infos = reset_out, {}
    done = False
    ep_reward = 0.0
    steps = 0
    while not done:
      state = self._global_state(env, obs_dict, infos)
      action_dict = self._select_actions(obs_dict, epsilon)
      step_out = env.step(action_dict)
      if len(step_out) == 5:
        next_obs, rewards, terminateds, truncateds, infos = step_out
        done = bool(
          terminateds.get("__all__", False) or truncateds.get("__all__", False)
        )
      else:
        next_obs, rewards, dones, infos = step_out
        done = bool(dones.get("__all__", False))

      reward = float(rewards[self._agents[0]])
      ep_reward += reward
      steps += 1

      obs_n = np.stack(
        [flatten_obs(obs_dict[a], self._obs_space) for a in self._agents],
        axis=0,
      ).astype(np.float32)
      actions = np.asarray(
        [int(action_dict[a]) for a in self._agents], dtype=np.int64
      )
      if not done:
        next_obs_n = np.stack(
          [flatten_obs(next_obs[a], self._obs_space) for a in self._agents],
          axis=0,
        ).astype(np.float32)
        next_state = self._global_state(env, next_obs, infos)
      else:
        next_obs_n = np.zeros_like(obs_n)
        next_state = state.copy()

      self._buffer.push((
        obs_n,
        actions,
        np.float32(reward),
        next_obs_n,
        self._fit_state(state),
        self._fit_state(next_state),
        np.float32(done),
      ))
      obs_dict = next_obs
    return ep_reward, steps

  def _train_step(self):
    cfg = self.config
    batch = self._buffer.sample(cfg.train_batch_size)
    obs, actions, rewards, next_obs, states, next_states, dones = batch

    obs_t = torch.as_tensor(obs, device=self._device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=self._device)
    rewards_t = torch.as_tensor(rewards, device=self._device)
    next_obs_t = torch.as_tensor(next_obs, device=self._device)
    states_t = torch.as_tensor(states, device=self._device)
    next_states_t = torch.as_tensor(next_states, device=self._device)
    dones_t = torch.as_tensor(dones, device=self._device)

    q_taken = []
    for i, net in enumerate(self.q_nets):
      q = net(obs_t[:, i, :])
      q_taken.append(q.gather(1, actions_t[:, i : i + 1]).squeeze(1))
    agent_q = torch.stack(q_taken, dim=1)
    q_tot = self.mixer(agent_q, states_t)
    if q_tot.ndim > 1:
      q_tot = q_tot.reshape(agent_q.shape[0])

    with torch.no_grad():
      next_list = []
      for i, net in enumerate(self.target_q_nets):
        next_list.append(net(next_obs_t[:, i, :]).max(dim=1).values)
      next_agent_q = torch.stack(next_list, dim=1)
      next_q_tot = self.target_mixer(next_agent_q, next_states_t)
      if next_q_tot.ndim > 1:
        next_q_tot = next_q_tot.reshape(next_agent_q.shape[0])
      target = rewards_t + float(cfg.gamma) * (1.0 - dones_t) * next_q_tot

    loss = F.mse_loss(q_tot, target)
    self._optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
      list(self.q_nets.parameters()) + list(self.mixer.parameters()),
      float(cfg.grad_clip),
    )
    self._optimizer.step()
    return float(loss.item())

  @override(Algorithm)
  def training_step(self):
    self._init_value_decomp_modules()
    self._train_iteration += 1
    cfg = self.config
    epsilon = self._epsilon()

    local = self.workers.local_worker()
    for policy in local.policy_map.values():
      if hasattr(policy, "set_epsilon"):
        policy.set_epsilon(epsilon)

    env = _unwrap_env(local.env)
    ep_rewards = []
    ep_lens = []
    for _ in range(int(cfg.episodes_per_iter)):
      rew, steps = self._run_episode(env, epsilon)
      ep_rewards.append(rew)
      ep_lens.append(steps)

    losses = []
    if len(self._buffer) >= int(cfg.train_batch_size):
      for _ in range(int(cfg.gradient_steps_per_iter)):
        losses.append(self._train_step())

    if self._train_iteration % int(cfg.target_network_update_freq) == 0:
      self.target_q_nets = copy.deepcopy(self.q_nets)
      self.target_mixer.load_state_dict(self.mixer.state_dict())

    self._sync_weights_to_policies()

    mean_rew = float(np.mean(ep_rewards)) if ep_rewards else 0.0
    mean_len = float(np.mean(ep_lens)) if ep_lens else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0
    self._last_vdn_metrics = {
      "episode_reward_mean": mean_rew,
      "episode_len_mean": mean_len,
      "mean_td_loss": mean_loss,
      "epsilon": epsilon,
      "replay_size": len(self._buffer),
      "mixer": str(cfg.mixer),
    }
    return {
      "episode_reward_mean": mean_rew,
      "episode_len_mean": mean_len,
      "custom_metrics": {
        "vdn_qmix_episode_reward_mean": mean_rew,
        "vdn_qmix_td_loss": mean_loss,
        "vdn_qmix_epsilon": epsilon,
      },
      "info": {
        "learner": {
          "default_policy": {
            "mean_td_loss": mean_loss,
            "epsilon": epsilon,
            "replay_size": len(self._buffer),
            "mixer": str(cfg.mixer),
          }
        }
      },
    }

  @override(Algorithm)
  def __getstate__(self):
    state = super().__getstate__()
    if self._vdn_initialized:
      state["vdn_qmix"] = {
        "q_nets": [n.state_dict() for n in self.q_nets],
        "mixer": self.mixer.state_dict(),
        "train_iteration": self._train_iteration,
        "agents": self._agents,
        "obs_dim": self._obs_dim,
        "n_actions": self._n_actions,
        "state_dim": self._state_dim,
      }
    return state

  @override(Algorithm)
  def __setstate__(self, state):
    super().__setstate__(state)
    payload = state.get("vdn_qmix")
    if not payload:
      return
    self._init_value_decomp_modules()
    for net, sd in zip(self.q_nets, payload["q_nets"]):
      net.load_state_dict(sd)
    self.mixer.load_state_dict(payload["mixer"])
    self.target_q_nets = copy.deepcopy(self.q_nets)
    self.target_mixer = copy.deepcopy(self.mixer)
    self._train_iteration = int(payload.get("train_iteration", 0))
    self._sync_weights_to_policies()


class VDNConfig(VDNQMIXConfig):
  def __init__(self, algo_class=None):
    super().__init__(algo_class=algo_class or VDN)
    self.mixer = "vdn"


class VDN(VDNQMIX):
  @classmethod
  def get_default_config(cls):
    return VDNConfig()


class QMIXConfig(VDNQMIXConfig):
  def __init__(self, algo_class=None):
    super().__init__(algo_class=algo_class or QMIX)
    self.mixer = "qmix"


class QMIX(VDNQMIX):
  @classmethod
  def get_default_config(cls):
    return QMIXConfig()
