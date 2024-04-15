from src.utilities import load_config_file, write_config_file
from src.generators_factory import ACGfactory
from src.environment import BaseEnvironment

from ray.rllib.algorithms.algorithm import Algorithm
from typing import Tuple
import json
import os


class Algorithm:
  def __init__(self, algo: str):
    self.algo_config_generator = ACGfactory.create(algo)

  def build(
      self, 
      environment: BaseEnvironment,
      config_dir: str
    ) -> Algorithm:
    # load the configuration files from the provided directory
    env_config, ray_config, exp_config = self.load_config_files(config_dir)
    # generate `AlgorithmConfig`
    algo_config = self.algo_config_generator.generate_algo_config(
      environment, env_config, ray_config, exp_config
    )
    # build `Algorithm`
    algo = algo_config.build()
    # save the configuration files
    self.write_config_files(env_config, exp_config, algo.logdir)
    self.print_algo_config(algo)
    return algo
  
  def print_algo_config(self, algo: Algorithm, to_file: bool = True):
    """
    Print the `AlgorithmConfig` in json format (by default, to a file saved 
    in the `Algorithm` logdir)
    """
    jj = self.algo_config_generator.to_json(algo.config)
    if to_file:
      write_config_file(
        jj, os.path.join(algo.logdir, "complete_config"), "ray_config.json"
      )
    else:
      print(jj)
  
  def load_config_files(self, config_dir: str) -> Tuple[dict, dict, dict]:
    """
    Load the configuration files needed to build the `Algorithm`
    """
    # read environment configuration file (mandatory!)
    env_config = load_config_file(os.path.join(config_dir, "env_config.json"))
    if env_config is None:
      raise FileNotFoundError(f"env_config.json not found in {config_dir}")
    # read ray configuration file
    ray_config = load_config_file(os.path.join(config_dir, "ray_config.json"))
    # read experiment configuration file
    exp_config = load_config_file(os.path.join(config_dir, "exp_config.json"))
    return env_config, ray_config, exp_config
  
  def write_config_files(
      self, env_config: dict, exp_config: dict, logdir: str
    ):
    """
    Write the environment and experiment configuration files into the 
    algorithm logdir
    """
    # write environment configuration file
    write_config_file(
      json.dumps(env_config), 
      os.path.join(logdir, "complete_config"), 
      "env_config.json"
    )
    # write experiment configuration file
    if env_config is not None:
      write_config_file(
        json.dumps(exp_config), 
        os.path.join(logdir, "complete_config"), 
        "exp_config.json"
      )

