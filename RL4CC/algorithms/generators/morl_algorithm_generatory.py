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
from RL4CC.environment.base_multiobjective_environment import (
  BaseMultiObjectiveEnvironment
)
from RL4CC.log_and_report.rl4cc_logger import Logger

from abc import ABC, abstractmethod
import mo_gymnasium as mo_gym
from copy import deepcopy
from typing import Tuple


class MORLAlgorithmGenerator(ABC):
  backend = "morl"
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ) -> None:
    self.logger = logger
    self.algo = None
    self._algo_init_keys = []
    self._protected_keys = [
      # (key, key group)
      ("rollout_fragment_length", "rollouts"),
      ("batch_mode", "rollouts"),
      ("train_batch_size", "training"),
      ("evaluation_interval", "evaluation"),
      ("evaluation_duration", "evaluation"),
      ("env", "environment"),
      ("env_config", "environment")
    ]
    self._suggested_keys = [
      # (key, key group, tunable)
      ("duration_per_worker",  "rollouts", "not_tunable"),
      ("duration_unit",  "rollouts", "not_tunable"),
      ("batch_size",  "training", "not_tunable"),
      ("num_train_batches",  "training", "not_tunable")
      ("evaluation_duration_per_worker", "evaluation", "not_tunable")
    ]
  
  @abstractmethod
  def convert_exploration_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to exploration behavior
    """
    pass

  @abstractmethod
  def convert_evaluation_parameters(
      self, all_params: dict, env_config: dict, eval_interval: int
    ):
    """
    Defines the appropriate parameters related to exploration behavior
    """
    pass
  
  @abstractmethod
  def convert_rollout_parameters(self, all_params: dict, env_config: dict):
    """
    Defines the appropriate parameters related to the definition and
    behavior of rollout workers, according to the provided keys
    """
    pass

  @abstractmethod
  def convert_training_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to the definition and 
    behavior of the policy training algorithm, according to the provided keys
    """
    pass

  @abstractmethod
  def generate_algo(
      self,
      env_config: dict,
      algo_config: dict = None,
      exp_logdir: str = None,
      eval_interval: int = None
    ):
    """
    Generates the `MORL-Baselines` algorith considering the provided 
    environment and configuration dictionaries
    """
    pass
  
  def make_env(
      self, 
      env_config: dict, 
      eval_env_config: dict = None, 
      exp_logdir: str = None
    ) -> Tuple[BaseMultiObjectiveEnvironment, BaseMultiObjectiveEnvironment]:
    # training environment
    updated_env_config = deepcopy(env_config)
    updated_env_config["exp_logdir"] = exp_logdir
    env = mo_gym.make(env_config["env_name"], env_config = updated_env_config)
    # evaluation environment
    updated_eval_env_config = deepcopy(eval_env_config)
    updated_eval_env_config["exp_logdir"] = exp_logdir
    eval_env = mo_gym.make(
      env_config["env_name"], env_config = updated_eval_env_config
    )
    return env, eval_env
  
  def process_config_parameters(
      self, algo_config: dict, env_config: dict, eval_interval: int
    ) -> Tuple[dict, dict]:
    dict_to_keep = {}
    dict_to_drop = {}
    if algo_config is not None:
      # check the presence of protected/suggested keys
      (
        using_suggested_keys, using_protected_keys
      ) = AlgoConfigGenerator._validate_key_usage(
        self._protected_keys, self._suggested_keys, algo_config, self.logger
      )
      # convert parameters
      updated_algo_config = deepcopy(algo_config)
      self.convert_exploration_parameters(updated_algo_config)
      self.convert_rollout_parameters(updated_algo_config, env_config)
      self.convert_training_parameters(updated_algo_config)
      self.convert_evaluation_parameters(
        updated_algo_config, env_config, eval_interval
      )
      # filter
      for k,v in updated_algo_config.items():
        if k in self.algo_init_keys:
          dict_to_keep[k] = v
        else:
          dict_to_drop[k] = v
    return dict_to_keep, dict_to_drop
