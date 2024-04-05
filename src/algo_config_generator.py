"""
Copyright 2024 Federica Filippini

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
from src.environment import BaseEnvironment

from ray.rllib.algorithms import AlgorithmConfig
from ray.tune.registry import get_trainable_cls
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple
import os


class AlgoConfigGenerator(ABC):
  def __init__(self):
    self.algo = None
  
  def generate_algo_config(
      self, 
      environment: BaseEnvironment,
      env_config: dict,
      ray_config: dict = None,
      exp_config: dict = None
    ) -> AlgorithmConfig:
    """
    Define the `AlgorithmConfig` considering the provided environment and 
    configuration dictionaries
    """
    algo_config = (
      get_trainable_cls(self.algo)
      .get_default_config()
      # environment
      .environment(environment, env_config=env_config)
    )
    # set ray config parameters
    if ray_config is not None:
      # resources
      algo_config.resources(**ray_config["resources"])
      # framework
      algo_config.framework(ray_config["framework"])
      # rollouts
      algo_config.rollouts(**ray_config["rollouts"]) 
      # reporting
      algo_config.reporting(**ray_config["reporting"])
      # training
      algo_config.training(**ray_config["training"])
      # exploration
      api = ray_config.get("rl_module", {}).get("_enable_rl_module_api", False)
      if not api:
        algo_config.exploration(**ray_config["exploration"])
    # if a logdir is provided, set it (default: ~/ray_results/<experiment>)
    debugging_config = {}
    if exp_config is not None and "logdir" in exp_config:
      now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
      exp_logdir = os.path.join(
        exp_config["logdir"], f"{self.algo}_{environment.name}_{now}"
      )
      os.makedirs(exp_logdir, exist_ok=True)
      # extract base debugging configuration...
      if ray_config is not None and "debugging" in ray_config:
        debugging_config = ray_config["debugging"]
      # ...and update it
      if "logger_config" not in debugging_config:
        debugging_config["logger_config"] = {}
      debugging_config["logger_config"]["logdir"] = exp_logdir
    # debugging
    if len(debugging_config) > 0:
      algo_config.debugging(**debugging_config)
    return algo_config
  
  @abstractmethod
  def get_default_config(self) -> Tuple[dict, dict]:
    pass

##############################################################################
# PPO
##############################################################################
class PPOConfigGenerator(AlgoConfigGenerator):
  def __init__(self):
    super().__init__()
    self.algo = "PPO"
    

##############################################################################
# DQN
##############################################################################
class DQNConfigGenerator(AlgoConfigGenerator):
  def __init__(self):
    super().__init__()
    self.algo = "DQN"
