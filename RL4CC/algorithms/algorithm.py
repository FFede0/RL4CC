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
try:
  from RL4CC.algorithms.backends import MORLAlgorithmBackend
except ImportError:
  pass
try:
  from RL4CC.algorithms.backends import RayAlgorithmBackend
except ImportError:
  pass
from RL4CC.algorithms.generators_factory import AGfactory
from RL4CC.utilities.common import write_config_file
from RL4CC.log_and_report.rl4cc_logger import Logger

from ray.rllib.algorithms.algorithm import Algorithm as RayAlgorithm
from ray.rllib.algorithms import AlgorithmConfig
import os


class Algorithm:
  def __init__(
      self, 
      algo_name: str,
      checkpoint_path: str = None,
      env_config: dict = None,
      learner_config: dict = None,
      logdir: str = None,
      eval_interval: int = None,
      use_tune: bool = False,
      multiagent: bool = False,
      logger: Logger = Logger(name="RL4CC-Algorithm")
    ):
    self.logger = logger
    self.generator = AGfactory.create(
      algo_name, logger = self.logger
    )
    # switch according to backend
    if self.generator.backend == "ray":
      self.backend = RayAlgorithmBackend(
        generator = self.generator,
        checkpoint_path = checkpoint_path,
        env_config = env_config,
        learner_config = learner_config,
        logdir = logdir,
        eval_interval = eval_interval,
        use_tune = use_tune,
        multiagent = multiagent,
        logger = self.logger
      )
    elif self.generator.backend == "morl":
      self.backend = MORLAlgorithmBackend(
        generator = self.generator,
        checkpoint_path = checkpoint_path,
        env_config = env_config,
        learner_config = learner_config,
        logdir = logdir,
        eval_interval = eval_interval,
        logger = self.logger
      )
    else:
      raise ValueError(
        f"Unsupported algorithm backend: {self.generator.backend}"
      )
    self.logdir = self.backend.logdir
 
  def build(self):
    """
    Build the `Algorithm` according to the provided checkpoint path or 
    configuration dictionaries
    """
    self.backend.build()
    self.logdir = self.backend.logdir
  
  def compute_single_action(self, obs, explore: bool = False, weight = None):
    """
    Compute a single action from agent(s) that received an observation
    """
    return self.backend.compute_single_action(
      obs,
      explore = explore,
      weight = weight
    )

  def get_policy(self, policy_id: str = None):
    """
    Get the policy of the `Algorithm`
    """
    if self.generator.backend != "ray":
      raise NotImplementedError
    pid = policy_id if policy_id is not None else DEFAULT_POLICY_ID
    return self.backend.get_policy(pid)
  
  def get_weights(self, policy_ids: list = None):
    """
    Get a dictionary of weights for the given `Algorithm` policies (returns 
    the weights for all policies if `policy_ids` is None)
    ---
    Note: for a single-agent scenario with the default policy, the key for 
    the weights dictionary is `default_policy`
    """
    if self.generator.backend != "ray":
      raise NotImplementedError
    return self.backend.get_weights(policy_ids)
  
  def set_weights(self, weights: dict):
    """
    Set the weights of the `Algorithm` policies
    """
    if self.generator.backend != "ray":
      raise NotImplementedError
    return self.algo.set_weights(weights)
  
  def load_checkpoint(
      self, 
      path: str, 
      policy_ids: list = None,
      policy_mapping_fn = None,
      policies_to_train: list = None
    ):
    """
    Load the provided `Algorithm` checkpoint
    """
    if (not os.path.exists(path) or not os.path.isdir(path)):
      raise FileNotFoundError(
        f"ERROR: checkpoint path {path} does not exist or is invalid"
      )
    # check if the checkpoint is a manual or automatic checkpoint
    if os.path.exists(os.path.join(path, "MANUAL_CHECKPOINT")):
      self._load_manual_checkpoint(path)
    else:
      self.algo = RayAlgorithm.from_checkpoint(
        path,
        policy_ids = policy_ids,
        policy_mapping_fn = policy_mapping_fn,
        policies_to_train = policies_to_train
      )
    self.algo_config = self.algo.config
    self.logdir = self.algo.logdir
    self.logger.warn(
      f"Algorithm restored from checkpoint; output directory: {self.logdir}"
    )
  
  def train(self) -> dict:
    """
    Perform one training iteration
    """
    return self.backend.train()
  
  def evaluate(self) -> dict:
    """
    Perform one evaluation step
    """
    return self.backend.evaluate()
  
  def stop(self) -> dict:
    """
    Releases all resources used by this trainable
    """
    self.backend.stop()
  
  def last_iteration(self) -> int:
    return self.backend.last_iteration()
  
  def save_checkpoint(self, manual: bool = False, path: str = None) -> str:
    """
    Save an `Algorithm` checkpoint (the checkpoint directory name is given 
    by the last iteration number); provide the parameter `manual = True` if 
    the checkpoint should be manually generated instead of relying on the 
    Ray RLLib implementation of `algo.save()`
    """
    return self.backend.save_checkpoint(manual = manual, path = path)
  
  def print_algo_config(self, to_file: bool = True):
    """
    Print the `AlgorithmConfig` in json format (by default, to a file saved 
    in the `Algorithm` logdir)
    """
    jj = self.generator.to_json(self.backend.algo_config)
    if to_file:
      write_config_file(
        jj,
        os.path.join(self.logdir, "complete_config"),
        "learner_config.json"
      )
    else:
      print(jj)
