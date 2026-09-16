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
from RL4CC.log_and_report.rl4cc_logger import Logger
from RL4CC.algorithms.generators import (
  AlgoConfigGenerator,
  MORLAlgorithmGenerator
)

from morl_baselines.multi_policy.envelope.envelope import Envelope
from copy import deepcopy
import numpy as np
import os


class EnvelopeQLearningGenerator(MORLAlgorithmGenerator):
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ) -> None:
    super().__init__(logger)
    self.algo = "EnvelopeQLearning"
    self._algo_init_keys = [
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
  
  def convert_evaluation_parameters(
      self, all_params: dict, env_config: dict, eval_interval: int
    ):
    # evaluation interval
    if eval_interval is not None and not np.isinf(eval_interval):
      all_params["eval_freq"] = eval_interval * all_params["total_timesteps"]
    eval_config = all_params.pop("evaluation", {})
    # env config for evaluation
    eval_env_config = deepcopy(env_config)
    if "evaluation_config" in eval_config:
      eval_env_config.update(eval_config["evaluation_config"])
    all_params["evaluation_config"] = eval_env_config
    # number of workers
    if eval_config.pop("evaluation_num_workers", 1) != 1:
      raise ValueError(
        "MORL algorithms currently support only a single rollout worker"
      )
    # duration
    unit = eval_config.pop("evaluation_duration_unit", "episodes")
    if "evaluation_duration_per_worker" in eval_config:
      duration = eval_config.pop(
        "evaluation_duration_per_worker"
      )
      if unit == "timesteps":
        nspe = AlgoConfigGenerator.compute_num_steps_per_episode(env_config)
        all_params["num_eval_episodes_for_front"] = int(np.ceil(duration/nspe))
      elif unit == "episodes":
        all_params["num_eval_episodes_for_front"] = duration
      else:
        raise ValueError(f"ERROR: invalid `evaluation_duration_unit` {unit}")
    # all additional evaluation parameters
    for k, v in eval_config.items():
      all_params[k] = v
  
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
      if k in self._algo_init_keys:
        all_params[k] = v
      else:
        # gradient clip
        if k == "grad_clip":
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
        # reference points
        elif k == "ref_point":
          all_params["ref_point"] = np.array(v)
        # replay buffer
        elif k == "replay_buffer_config":
          # -- capacity
          if "capacity" in v:
            all_params["buffer_size"] = int(v["capacity"])
          # -- prioritize experience replay
          if "prioritized_replay_alpha" in v:
            all_params["per"] = True
            all_params["per_alpha"] = v["prioritized_replay_alpha"]
        # target network update frequency
        elif k == "target_network_update_freq":
          all_params["target_net_update_freq"] = v
        else:
          all_params[k] = v
  
  def generate_algo(
      self,
      env_config: dict,
      algo_config: dict = None,
      exp_logdir: str = None,
      eval_interval: int = None
    ):
    """
    Generates the `MORL-Baselines::Envelope` algorith considering the provided 
    environment and configuration dictionaries
    """
    # process configuration parameters
    morl_config, all_params = self.process_config_parameters(
      algo_config, env_config, eval_interval
    )
    # make environment
    env, eval_env = self.make_env(
      env_config, 
      all_params.pop("evaluation_config"), 
      exp_logdir
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
    return algo, eval_env, all_params
