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
from RL4CC.algorithms.generators.ppo_config_generator import PPOConfigGenerator
from RL4CC.log_and_report.rl4cc_logger import Logger

from ray.rllib.algorithms import AlgorithmConfig


class MAPPOConfigGenerator(PPOConfigGenerator):
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ):
    super().__init__(logger)
    self.algo = "MAPPO"
    # generate default `AlgorithmConfig`
    self.generate_default_config()
  
  def generate_algo_config(
      self,
      env_config: dict,
      ray_config: dict = None,
      exp_logdir: str = None,
      eval_interval: int = None,
      use_tune: bool = False,
      multiagent: bool = False
    ) -> AlgorithmConfig:
    """
    Defines the `AlgorithmConfig` considering the provided environment and
    configuration dictionaries
    ---
    MAPPO is a multi-agent algorithm by definition
    """
    return super().generate_algo_config(
      env_config,
      ray_config = ray_config,
      exp_logdir = exp_logdir,
      eval_interval = eval_interval,
      use_tune = use_tune,
      multiagent = True
    )
  
  def process_config_parameters(
      self,
      ray_config: dict,
      env_config: dict,
      exp_logdir: str = None,
      eval_interval: int = None
    ) -> dict:
    """
    Processes the configuration parameters, extracting the relevant
    information to define an `AlgorithmConfig`
    ---
    MAPPO exploits a centralized critic model by default
    """
    all_params = super().process_config_parameters(
      ray_config, 
      env_config, 
      exp_logdir = exp_logdir, 
      eval_interval = eval_interval
    )
    if "model" not in all_params:
      all_params["model"] = {}
    if "custom_model" not in all_params["model"]:
      all_params["model"]["custom_model"] = "centralizedcritic"
    return all_params
