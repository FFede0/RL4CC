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
from RL4CC.utilities.common import NumpyEncoder, json_to_array_dict
from RL4CC.algorithms.generators_factory import ACGfactory
from RL4CC.utilities.common import write_config_file
from RL4CC.log_and_report.rl4cc_logger import Logger

from ray.rllib.policy.sample_batch import (
  DEFAULT_POLICY_ID,
  SampleBatch, 
  MultiAgentBatch
)
from ray.rllib.algorithms.algorithm import Algorithm as RayAlgorithm
from ray.rllib.algorithms import AlgorithmConfig
import cloudpickle
import json
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
      multiagent: bool = False,
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
        use_tune = use_tune,
        multiagent = multiagent
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
  
  def compute_single_action(self, obs, explore: bool = False):
    """
    Compute a single action from agent(s) that received an observation
    """
    action = None
    if isinstance(obs, dict):
      action = {}
      for agent in obs:
        action[agent] = self.get_policy(agent).compute_single_action(
          obs[agent], explore = explore
        )[0]
    else:
      action = self.get_policy().compute_single_action(
        obs, explore = explore
      )[0]
    return action

  def get_policy(self, policy_id: str = None):
    """
    Get the policy of the `Algorithm`
    """
    pid = policy_id if policy_id is not None else DEFAULT_POLICY_ID
    return self.algo.get_policy(pid)
  
  def get_weights(self, policy_ids: list = None):
    """
    Get a dictionary of weights for the given `Algorithm` policies (returns 
    the weights for all policies if `policy_ids` is None)
    ---
    Note: for a single-agent scenario with the default policy, the key for 
    the weights dictionary is `default_policy`
    """
    return self.algo.get_weights(policy_ids)
  
  def set_weights(self, weights: dict):
    """
    Set the weights of the `Algorithm` policies
    """
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
  
  def save_checkpoint(self, manual: bool = False, path: str = None) -> str:
    """
    Save an `Algorithm` checkpoint (the checkpoint directory name is given 
    by the last iteration number); provide the parameter `manual = True` if 
    the checkpoint should be manually generated instead of relying on the 
    Ray RLLib implementation of `algo.save()`
    """
    checkpoint_dir = path if path is not None else os.path.join(
      self.algo.logdir, f"checkpoints/{self.last_iteration()}"
    )
    last_checkpoint_dir = None
    # automatic checkpoint
    if not manual:
      save_result = self.algo.save(checkpoint_dir = checkpoint_dir)
      last_checkpoint_dir = save_result.checkpoint.path
    else:
      # manual checkpoint
      os.makedirs(checkpoint_dir, exist_ok = True)
      # -- algorithm state
      algo_state = {
        "algorithm_class": self.algo.__class__,
        "config": self.algo.config.to_dict(),
        "state": self.algo.__getstate__()
      }
      # -- replay buffer
      replay_buffer = getattr(self.algo, "local_replay_buffer", None)
      if replay_buffer:
        algo_state["replay_buffer_state"] = replay_buffer.get_state()
      # -- save
      with open(os.path.join(checkpoint_dir, "algo_state.pkl"), "wb") as f:
        cloudpickle.dump(algo_state, f)
      # -- weights
      weights = self.get_weights()
      with open(os.path.join(checkpoint_dir, "weights.json"), "w") as ost:
        json.dump(
          weights, ost, indent = 2, cls = NumpyEncoder, sort_keys = True
        )
      # -- "manual checkpoint" indicator (for loading)
      with open(os.path.join(checkpoint_dir, "MANUAL_CHECKPOINT"), "wb") as f:
        pass
      last_checkpoint_dir = checkpoint_dir
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
  
  def _load_manual_checkpoint(self, path: str):
    """
    Load a manually-saved checkpoint (algo state, weights, replay buffer)
    """
    # load algorithm state
    checkpoint_file = os.path.join(path, "algo_state.pkl")
    with open(checkpoint_file, "rb") as f:
      self.logger.log(f"Loading algorithm state from: {checkpoint_file}", 1)
      algo_state = cloudpickle.load(f)
    algo_cls = algo_state["algorithm_class"]
    state = algo_state["state"]
    # -- get configuration
    config = algo_state["config"]
    config["create_env_on_driver"] = False
    config["disable_env_checking"] = True
    config["callbacks"] = None
    # create algorithm
    self.algo = algo_cls(config = config)
    # setup all algorithm components (including replay buffer)
    self.algo.setup(config = config)
    # -- state
    self.algo.__setstate__(state)
    # -- policy target (if any)
    for agent in self.algo.config.policies:
      try:
        self.get_policy(agent).update_target()
      except Exception as e:
        self.logger.warn(f"update_target not available for {agent} policy")
    # -- model weights
    self.set_weights(
      json_to_array_dict(os.path.join(path, "weights.json"))
    )
    self.logger.log(
      "Algorithm state and policy weights loaded successfully.", 2
    )
    # -- replay buffer (if any)
    if "replay_buffer_state" in algo_state:
      try:
        old_state = algo_state["replay_buffer_state"]
        if self.algo.local_replay_buffer is not None:
          # wrap SampleBatches as MultiAgentBatches
          if "_storage" in old_state and isinstance(old_state["_storage"],list):
            wrapped = []
            for sample in old_state["_storage"]:
              if isinstance(sample, SampleBatch):
                wrapped.append(MultiAgentBatch(
                  {"default_policy": sample}, sample.count)
                )
              elif isinstance(sample, MultiAgentBatch):
                wrapped.append(sample)
              else:
                raise ValueError(f"Unexpected batch type: {type(sample)}")
            old_state["_storage"] = wrapped
          self.algo.local_replay_buffer.set_state(old_state)
      except Exception as e:
        raise RuntimeError(
          f"Warning: Replay buffer could not be restored: {e}"
        )
