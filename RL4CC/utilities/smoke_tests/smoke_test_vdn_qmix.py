"""Local smoke test for VDN / QMIX (a few train iterations)."""
import os
import tempfile
import traceback

import numpy as np
from gymnasium.spaces import Box, Discrete, Dict
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env

import RL4CC.algorithms  # register algos
from RL4CC.environment.base_multiagent_environment import BaseMultiAgentEnvironment
from RL4CC.algorithms.algorithm import Algorithm as RL4CCAlgorithm


class SmokeTeamEnv(BaseMultiAgentEnvironment):

  def define_observation_spaces(self):
    single = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    self.single_agent_observation_space = single
    self._observation_space = Dict({a: single for a in self.agents})

  def define_action_spaces(self):
    single = Discrete(3)
    self.single_agent_action_space = single
    self._action_space = Dict({a: single for a in self.agents})

  def reset(self, *, seed=None, options=None):
    self.current_time = self.min_time
    self._step_i = 0
    self._state = np.zeros(6, dtype=np.float32)
    obs = {
      a: np.zeros(4, dtype=np.float32) for a in self.agents
    }
    infos = {"__common__": {"global_state": self._state.copy()}}
    return obs, infos

  def build_global_state(self):
    return self._state.copy()

  def _global_state_dim(self):
    return 6

  def step(self, action_dict):
    self._step_i += 1
    self.current_time += self.time_step
    # Shared reward: prefer action 1.
    reward = float(sum(1.0 if int(action_dict[a]) == 1 else -0.1 for a in self.agents))
    self._state = np.array(
      [self._step_i / 10.0, reward / 3.0, 0, 0, 0, 0], dtype=np.float32
    )
    obs = {
      a: np.array([self._step_i / 10.0, float(action_dict[a]) / 2.0, 0, 0], dtype=np.float32)
      for a in self.agents
    }
    terminated = self.current_time >= self.max_time
    terminateds = {a: terminated for a in self.agents}
    terminateds["__all__"] = terminated
    truncateds = {a: False for a in self.agents}
    truncateds["__all__"] = False
    rewards = {a: reward for a in self.agents}
    infos = {"__common__": {"global_state": self._state.copy()}}
    return obs, rewards, terminateds, truncateds, infos


def _env_creator(env_config):
  if not isinstance(env_config, EnvContext):
    env_config = EnvContext(env_config, worker_index=0, vector_index=0)
  return SmokeTeamEnv(env_config)


register_env("SmokeTeamEnv", _env_creator)


def run_algo(algo_name: str, n_iters: int = 3) -> None:
  logdir = tempfile.mkdtemp(prefix=f"smoke_{algo_name}_")
  agents = ["A1", "A2", "A3"]
  env_config = {
    "env_name": "SmokeTeamEnv",
    "agents": agents,
    "min_time": 0,
    "max_time": 5,
    "time_step": 1,
    "exp_logdir": logdir,
  }
  ray_config = {
    "framework": "torch",
    "resources": {
      "num_gpus_master": 0,
      "num_cpus_master": 1,
    },
    "rollouts": {
      "num_rollout_workers": 0,
    },
    "training": {
      "lr": 5e-4,
      "gamma": 0.99,
      "batch_size": 16,
      "episodes_per_iter": 2,
      "gradient_steps_per_iter": 5,
      "target_network_update_freq": 2,
      "replay_buffer_capacity": 1000,
      "epsilon_start": 1.0,
      "epsilon_end": 0.05,
      "epsilon_decay_iters": 10,
      "mixing_embed_dim": 16,
      "model": {
        "fcnet_hiddens": [32, 16],
        "fcnet_activation": "relu",
        "custom_model": "value_decomp_q",
      },
    },
  }
  print(f"\n=== SMOKE {algo_name} ===")
  wrapper = RL4CCAlgorithm(
    algo_name=algo_name,
    env_config=env_config,
    ray_config=ray_config,
    logdir=logdir,
    multiagent=True,
  )
  wrapper.build()
  for i in range(1, n_iters + 1):
    result = wrapper.algo.train()
    custom = result.get("custom_metrics") or {}
    local = getattr(wrapper.algo, "_last_vdn_metrics", {}) or {}
    rew = local.get("episode_reward_mean", custom.get("vdn_qmix_episode_reward_mean"))
    loss = local.get("mean_td_loss", custom.get("vdn_qmix_td_loss"))
    print(
      f"  iter {i}: reward_mean={rew} td_loss={loss} "
      f"epsilon={local.get('epsilon')}"
    )
    if rew is None:
      raise RuntimeError(f"{algo_name} iter {i}: missing reward metric")
  wrapper.algo.stop()
  print(f"=== SMOKE {algo_name} OK (logdir={logdir}) ===")


def main():
  os.environ.setdefault("RAY_DEDUP_LOGS", "0")
  errors = []
  for name in ("VDN", "QMIX"):
    try:
      run_algo(name, n_iters=3)
    except Exception as e:
      errors.append((name, e))
      print(f"=== SMOKE {name} FAILED: {e} ===")
      traceback.print_exc()
  if errors:
    raise SystemExit(1)
  print("\nAll smoke tests passed.")


if __name__ == "__main__":
  main()
