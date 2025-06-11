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
from RL4CC.algorithms.generators.algo_config_generator import AlgoConfigGenerator
from RL4CC.algorithms.generators.ppo_config_generator import PPOConfigGenerator
from RL4CC.algorithms.generators.dqn_config_generator import DQNConfigGenerator
from RL4CC.algorithms.generators.sac_config_generator import SACConfigGenerator



class AlgoConfigGeneratorsFactory:
  """
  Factory of `AlgoConfigGenerator`s
  """
  def __init__(self):
    self.algo_config_generators = {}
  
  def register(self, algo: str, generator: AlgoConfigGenerator):
    """
    Register the given `generator` under the provided `algo` name
    """
    self.algo_config_generators[algo] = generator
  
  def create(self, algo: str, **kwargs):
    """
    Create a new generator according to the given `algo` name
    """
    generator = self.algo_config_generators.get(algo)
    if not generator:
        raise ValueError(algo)
    return generator(**kwargs)


## Factory initialization
ACGfactory = AlgoConfigGeneratorsFactory()
ACGfactory.register("PPO", PPOConfigGenerator)
ACGfactory.register("DQN", DQNConfigGenerator)
ACGfactory.register("SAC", SACConfigGenerator)

