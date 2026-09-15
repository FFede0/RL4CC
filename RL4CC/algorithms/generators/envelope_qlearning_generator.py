"""
Copyright 2026 Federica Filippini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from RL4CC.algorithms.generators.algo_config_generator import (
  AlgoConfigGenerator
)
from RL4CC.algorithms.generators.morl_algorithm_generatory import (
  MORLAlgorithmGenerator
)
from RL4CC.log_and_report.rl4cc_logger import Logger

from morl_baselines.multi_policy.envelope.envelope import Envelope
from typing import Tuple
import os


class EnvelopeQLearningGenerator(MORLAlgorithmGenerator):
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ) -> None:
    super().__init__(logger)
    self.algo = "EnvelopeQLearning"
    self.algo_init_keys = [
      "batch_size",
      "buffer_size",
      "device",
      "envelope",
      "epsilon_decay_steps",
      "final_epsilon",
      "final_homotopy_lambda",
      "gamma",
      "gradient_updates",
      "initial_epsilon",
      "initial_homotopy_lambda",
      "homotopy_decay_steps",
      "learning_rate",
      "learning_starts",
      "max_grad_norm",
      "net_arch",
      "num_sample_w",
      "per",
      "per_alpha",
      "seed",
      "target_net_update_freq",
      "tau"
    ]
  
  def convert_exploration_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to exploration behavior
    """
    exploration_params = all_params.pop("exploration", {})
    if not exploration_params.get("explore", True):
      all_params["initial_epsilon"] = 0.0
      all_params["final_epsilon"] = 0.0
      all_params["epsilon_decay_steps"] = 0
    else:
      if "exploration_config" in exploration_params:
        all_params["initial_epsilon"] = exploration_params[
          "exploration_config"
        ]["initial_epsilon"]
        all_params["final_epsilon"] = exploration_params[
          "exploration_config"
        ]["final_epsilon"]
        all_params["epsilon_decay_steps"] = exploration_params[
          "exploration_config"
        ]["epsilon_timesteps"]
  
  def convert_rollout_parameters(self, all_params: dict, env_config: dict):
    """
    Defines the appropriate parameters related to the definition and
    behavior of rollout workers, according to the provided keys
    """
    rollout_params = all_params.pop("rollouts", {})
    if rollout_params.get("num_rollout_workers", 1) != 1:
      raise ValueError(
        "MORL algorithms currently support only a single rollout worker"
      )
    # -- counting timesteps
    duration_unit = rollout_params.get("duration_unit", "timesteps")
    if duration_unit == "timesteps":
      all_params["total_timesteps"] = rollout_params["duration_per_worker"]
    # -- couting whole episodes
    elif duration_unit == "episodes":
      all_params["total_episodes"] = rollout_params["duration_per_worker"]
      nspe = AlgoConfigGenerator.compute_num_steps_per_episode(env_config)
      all_params["total_timesteps"] = nspe * all_params["total_episodes"]
    else:
      raise ValueError(f"Unsupported `duration_unit`: {duration_unit}")
  
  def convert_training_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to the definition and 
    behavior of the policy training algorithm, according to the provided keys
    """
    training_params = all_params.pop("training", {})
    for k, v in training_params.items():
      if k in self.algo_init_keys:
        all_params[k] = v
      else:
        # batch size
        if k == "train_batch_size":
          all_params["batch_size"] = v
        # gradient clip
        elif k == "grad_clip":
          all_params["max_grad_norm"] = v
        # learning rate
        elif k == "lr":
          all_params["learning_rate"] = v
        # network architecture
        elif k == "hiddens":
          all_params["net_arch"] = v
        # number of gradient updates per iteration
        elif k == "num_train_batches":
          all_params["gradient_updates"] = v
        # number of steps to be sampled before training starts
        elif k == "num_steps_sampled_before_learning_starts":
          all_params["learning_starts"] = v
        # replay buffer
        elif k == "replay_buffer_config":
          # -- capacity
          if "capacity" in v:
            all_params["buffer_size"] = v["capacity"]
          # -- prioritize experience replay
          if "prioritized_replay_alpha" in v:
            all_params["per"] = True
            all_params["per_alpha"] = v["prioritized_replay_alpha"]
        # target network update frequency
        elif k == "target_network_update_freq":
          all_params["target_net_update_freq"] = v
  
  def generate_algo(
      self,
      env_config: dict,
      algo_config: dict = None,
      exp_logdir: str = None
    ):
    """
    Generates the `MORL-Baselines::Envelope` algorith considering the provided 
    environment and configuration dictionaries
    """
    # make environment
    env, eval_env = self.make_env(env_config, algo_config, exp_logdir)
    # filter keys in algorithm configuration
    morl_config, additional_params = self.filter_algo_config(
      algo_config, env_config
    )
    # generate algorithm
    expname = self.algo if exp_logdir is None else os.path.basename(exp_logdir)
    algo = Envelope(
      env,
      **morl_config,
      log = True,
      project_name = self.algo,
      experiment_name = expname
    )
    return algo, eval_env, additional_params
