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
from algorithms.generators_factory import ACGfactory
from utilities.common import write_config_file
from utilities.logger import Logger

from ray.rllib.algorithms.algorithm import Algorithm as RayAlgorithm
from ray.rllib.algorithms import AlgorithmConfig
from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
import os


class Algorithm:
  def __init__(
      self, 
      algo_name: str,
      checkpoint_path: str = None,
      env_config: dict = None,
      ray_config: dict = None,
      tune_config: dict = None,
      base_logdir: str = None,
      eval_interval: int = None, 
      logger: Logger = Logger(name="RL4CC-Algorithm")
    ):
    self.logger = logger
    self.algo_config_generator = ACGfactory.create(
      algo_name, logger = self.logger
    )
    # load the Ray `Algorithm` from a checkpoint (if provided)
    if checkpoint_path is not None:
      self.load_checkpoint(checkpoint_path)
    # otherwise...
    else:
      if env_config is None:
        raise RuntimeError(
          "ERROR: no environment configuration provided"
        )
      # ...generate `AlgorithmConfig`
      self.algo_config = self.algo_config_generator.generate_algo_config(
        ray_config=ray_config,
        env_config=env_config,
        tune_config=tune_config,
        eval_interval=eval_interval,
        base_logdir=base_logdir,
      )
      # ...build and save the Ray `Algorithm`
      # TODO (Mohanad): Uncomment the build method once the error is solved
      #self.build(self.algo_config)

    if tune_config is not None:
      self.tune_config = self.get_tune_parameters(tune_config=tune_config)
      self.run_config = self.get_run_parameters(tune_config=tune_config)

  def build(self, algo_config: AlgorithmConfig):
    """
    Build the `Algorithm` according to the provided checkpoint path or 
    configuration dictionaries
    """
    self.algo = algo_config.build()
    self.logdir = self.algo.logdir
    self.logger.warn(
      f"Algorithm created; output directory: {self.logdir}"
    )
  
  def load_checkpoint(self, path: str):
    """
    Load the provided `Algorithm` checkpoint
    """
    if (not os.path.exists(path) or not os.path.isdir(path)):
      raise FileNotFoundError(
        f"ERROR: checkpoint path {path} does not exist or is invalid"
      )
    self.algo = RayAlgorithm.from_checkpoint(path)
    self.logdir = self.algo.logdir
    self.logger.warn(
      f"Algorithm restored from checkpoint; output directory: {self.logdir}"
    )
  
  def train(self) -> dict:
    """
    Perform one training iteration
    """
    return self.algo.train()
  
  def evaluate(self) -> dict:
    """
    Perform one evaluation step
    """
    return self.algo.evaluate()
  
  def stop(self) -> dict:
    """
    Releases all resources used by this trainable
    """
    self.algo.stop()
  
  def last_iteration(self) -> int:
    return self.algo.iteration
  
  def save_checkpoint(self) -> str:
    """
    Save an `Algorithm` checkpoint (the checkpoint directory name is given 
    by the last iteration number)
    """
    save_result = self.algo.save(
      checkpoint_dir = os.path.join(
        self.algo.logdir, f"checkpoints/{self.last_iteration()}"
      )
    )
    last_checkpoint_dir = save_result.checkpoint.path
    self.logger.log(
      "an Algorithm checkpoint has been created inside directory: "
      f"'{last_checkpoint_dir}'", 1
    )
    return last_checkpoint_dir
  
  def print_algo_config(self, to_file: bool = True):
    """
    Print the `AlgorithmConfig` in json format (by default, to a file saved 
    in the `Algorithm` logdir)
    """
    jj = self.algo_config_generator.to_json(self.algo.config)
    if to_file:
      write_config_file(
        jj, 
        os.path.join(self.algo.logdir, "complete_config"), 
        "ray_config.json"
      )
    else:
      print(jj)

  #TODO (mohanad): move to base experiemnt
  def get_tune_parameters(self, tune_config):
    # Get a copy of the tune_config
    tune_config_dict = tune_config

    # Handle keys to pass as a parse the tuning dictionary as Key word arguments
    tune_config_dict["num_samples"] = tune_config_dict["num_tune_trials"]
    tune_config_dict.pop("use_tune")
    tune_config_dict.pop("num_tune_trials")

    # Handle search algorithm and scheduler to covert them to their respective tune objects
    if "search_algorithm" in tune_config_dict:
      search_algorithm = list(tune_config_dict.get("search_algorithm").keys())[0]
      search_algorithm_config = tune_config_dict.get("search_algorithm")
      if search_algorithm == "hyperopt_search":
        try:
          tune_config_dict["search_algorithm"] = HyperOptSearch(**search_algorithm_config)
        except Exception as e:
          raise KeyError("Parameters passed to the hyperopt search algorithm are invalid!")
      else:
        raise KeyError("You are trying to pass a search algorithm that is not supported")

    if "scheduler" in tune_config_dict:
      scheduler= list(tune_config_dict.get("scheduler").keys())[0]
      scheduler_config = tune_config_dict.get("scheduler")
      if scheduler == "asha_scheduler":
        try:
          tune_config_dict["scheduler"] = ASHAScheduler(**scheduler_config)
        except:
          raise KeyError("Parameters passed to the ASHAScheduler scheduler are invalid!")
      else:
        raise KeyError("You are trying to pass a scheduler that is not supported")

    tune_params = tune.TuneConfig(**tune_config_dict)
    return tune_params





