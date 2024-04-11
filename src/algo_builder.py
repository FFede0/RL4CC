from src.generators_factory import ACGfactory
from src.environment import BaseEnvironment

from ray.rllib.algorithms.algorithm import Algorithm
import os


class AlgorithmBuilder:
  def __init__(self, algo: str):
    self.algo_config_generator = ACGfactory.create(algo)

  def build_algorithm(
      self, 
      environment: BaseEnvironment,
      env_config: dict,
      ray_config: dict = None,
      exp_config: dict = None
    ) -> Algorithm:
    # generate `AlgorithmConfig`
    algo_config = self.algo_config_generator.generate_algo_config(
      environment, env_config, ray_config, exp_config
    )
    # build `Algorithm`
    algo = algo_config.build()
    # save the configuration files
    self.print_algo_config(algo)
    return algo
  
  def print_algo_config(self, algo: Algorithm, to_file: bool = True):
    """
    Print the `AlgorithmConfig` in json format (by default, to a file saved 
    in the `Algorithm` logdir)
    """
    jj = self.algo_config_generator.to_json(algo.config)
    if to_file:
      config_dir = os.path.join(algo.logdir, "complete_config")
      os.makedirs(config_dir, exist_ok = True)
      with open(os.path.join(config_dir, "ray_config.json"), "w") as ostream:
        ostream.write(jj)
    else:
      print(jj)
