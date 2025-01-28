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
from RL4CC.algorithms.generators_factory import ACGfactory
from RL4CC.utilities.common import write_config_file
from RL4CC.utilities.logger import Logger

from ray.rllib.algorithms.algorithm import Algorithm as RayAlgorithm
from ray.rllib.algorithms import AlgorithmConfig
from ray.rllib.policy.policy import Policy
import os


class Algorithm:
  def __init__(
      self, 
      algo_name: str,
      checkpoint_path: str = None,
      env_config: dict = None,
      ray_config: dict = None,
      logdir: str = None,
      eval_interval: int = None,
      use_tune: bool = False,
      logger: Logger = Logger(name="RL4CC-Algorithm")
    ):
    self.logger = logger
    self.algo_config_generator = ACGfactory.create(
      algo_name, logger = self.logger
    )
    self.use_tune = use_tune
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
        ray_config = ray_config,
        env_config = env_config,
        eval_interval = eval_interval,
        exp_logdir = logdir,
        use_tune = use_tune
      )

  def build(self, algo_config: AlgorithmConfig = None):
    """
    Build the `Algorithm` according to the provided checkpoint path or 
    configuration dictionaries
    """
    if algo_config is not None:
      self.algo_config = algo_config
    self.algo = self.algo_config.build()
    self.logdir = self.algo.logdir
    self.logger.warn(
      f"`Algorithm` created; output directory: {self.logdir}"
    )

  def get_policy(self):
    """
    Get the policy of the `Algorithm`
    """
    return self.algo.get_policy()
  
  def load_checkpoint(self, path: str):
    """
    Load the provided `Algorithm` checkpoint
    """
    if (not os.path.exists(path) or not os.path.isdir(path)):
      raise FileNotFoundError(
        f"ERROR: checkpoint path {path} does not exist or is invalid"
      )
    self.algo = RayAlgorithm.from_checkpoint(path)
    self.algo_config = self.algo.config
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
    jj = self.algo_config_generator.to_json(self.algo_config)
    if to_file:
      write_config_file(
        jj,
        os.path.join(self.logdir, "complete_config"),
        "ray_config.json"
      )
    else:
      print(jj)
  
  def get_policy(self) -> Policy:
    return self.algo.get_policy()
