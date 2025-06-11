"""
Copyright 2024 Mohanad Diab, Federica Filippini

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
from RL4CC.algorithms.generators.dqn_config_generator import DQNConfigGenerator
from RL4CC.utilities.logger import Logger

from ray.rllib.algorithms import AlgorithmConfig


class SACConfigGenerator(DQNConfigGenerator):
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ):
    super().__init__(logger)
    self.algo = "SAC"
    # generate default `AlgorithmConfig`
    self.generate_default_config()
    # save a dictionary of algo methods and corresponding parameters
    self.save_algo_methods_dict()
  
  def validate_key_usage(self, all_params: dict):
    using_suggested_keys, using_protected_keys = super().validate_key_usage(
      all_params
    )
    if "model" in all_params:
      self.logger.warn(
        "SAC has two fields to configure for custom models: "
        "`policy_model_config` and `q_model_config`, the `model` field of the "
        "config is ignored"
      )
    return using_suggested_keys, using_protected_keys
  
  def to_dict(self, algo_config: AlgorithmConfig) -> dict:
    ray_config = super().to_dict(algo_config)
    # config dict for optimization (parameter and member have different names)
    if "optimization" in ray_config.get("not_classified", {}):
      opt = ray_config["not_classified"].pop("optimization")
      ray_config["training"]["optimization_config"] = opt
    return ray_config
