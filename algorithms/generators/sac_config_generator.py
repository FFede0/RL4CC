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
from algorithms.generators.dqn_config_generator import DQNConfigGenerator
from utilities.logger import Logger

from ray.rllib.algorithms.dqn.dqn import calculate_rr_weights
from ray.rllib.algorithms import AlgorithmConfig


class SACConfigGenerator(DQNConfigGenerator):
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ):
     
    self.algo = "SAC"
    super().__init__(logger)
    