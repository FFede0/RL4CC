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


class AlgoGeneratorsFactory:
  """
  Factory of Algorithms (Config) Generators
  """
  def __init__(self):
    self.algo_generators = {}
  
  def register(self, algo: str, generator):
    """
    Register the given `generator` under the provided `algo` name
    """
    self.algo_generators[algo] = generator
  
  def create(self, algo: str, **kwargs):
    """
    Create a new generator according to the given `algo` name
    """
    generator = self.algo_generators.get(algo)
    if not generator:
        raise ValueError(algo)
    return generator(**kwargs)


## Factory initialization
from RL4CC.log_and_report.rl4cc_logger import Logger
logger = Logger(name="RL4CC-AlgorithmGenerators")
AGfactory = AlgoGeneratorsFactory()
#
# -- Ray RLLib-based
try: 
  from RL4CC.algorithms.generators import (
    PPOConfigGenerator,
    DQNConfigGenerator,
    SACConfigGenerator,
    MAPPOConfigGenerator
  )
  AGfactory.register("PPO", PPOConfigGenerator)
  AGfactory.register("DQN", DQNConfigGenerator)
  AGfactory.register("SAC", SACConfigGenerator)
  AGfactory.register("MAPPO", MAPPOConfigGenerator)
except ImportError as error:
  logger.warn(
    f"Could not register algorithm: {error.msg!r}."
  )
#
# MORL-Baselines-based
try:
  from RL4CC.algorithms.generators import (
    EnvelopeQLearningGenerator
  )
  AGfactory.register("EnvelopeQLearning", EnvelopeQLearningGenerator)
except ImportError as error:
  logger.warn(
    f"Could not register algorithm: {error.msg!r}."
  )
