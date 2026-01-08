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


class MAPPOConfigGenerator(PPOConfigGenerator):
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ):
    super().__init__(logger)
    self.algo = "MAPPO"
    # generate default `AlgorithmConfig`
    self.generate_default_config()

