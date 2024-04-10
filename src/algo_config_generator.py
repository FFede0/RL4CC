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
from src.environment import BaseEnvironment

from ray.rllib.algorithms import AlgorithmConfig
from ray.tune.registry import get_trainable_cls
from abc import ABC, abstractmethod
from datetime import datetime
import inspect
import json
import os


class AlgoConfigGenerator(ABC):
  def __init__(self):
    self.algo = None
    self.base_algo_config = None
    self._protected_keys = [
      # (key, key group)
      ("rollout_fragment_length", "rollouts"),
      ("batch_mode", "rollouts"),
      ("train_batch_size", "training"),
      ("min_time_s_per_iteration", "reporting"),
      ("min_sample_timesteps_per_iteration", "reporting"),
      ("min_train_timesteps_per_iteration", "reporting"),
      ("num_gpus", "resources"),
      ("num_cpus_per_local_worker", "resources")
    ]
    self._suggested_keys = [
      # (key, key group)
      ("duration_per_worker",  "rollouts"),
      ("duration_unit",  "rollouts"),
      ("batch_size",  "training"),
      ("num_trained_batches",  "training"),
      ("num_gpus_master",  "resources"),
      ("num_cpus_master",  "resources")
    ]
    # save a dictionary of algo methods and corresponding parameters (useful 
    # to print the `AlgorithmConfig`)
    self.algo_methods = {}
    for f_name in dir(AlgorithmConfig):
      try:
        f = getattr(AlgorithmConfig, f_name)
        if callable(f) and not f_name.startswith("_"):
          f_info = inspect.getfullargspec(f)
          self.algo_methods[f_name] = set(f_info.args + f_info.kwonlyargs)
      except Exception as e:
        self.algo_methods[f_name] = {str(e)}
  
  def generate_default_config(self) -> AlgorithmConfig:
    """
    Generates the default `AlgorithmConfig` according to the class algorithm
    """
    self.base_algo_config = get_trainable_cls(self.algo).get_default_config()
  
  def generate_algo_config(
      self, 
      environment: BaseEnvironment,
      env_config: dict,
      ray_config: dict = None,
      exp_config: dict = None
    ) -> AlgorithmConfig:
    """
    Defines the `AlgorithmConfig` considering the provided environment and 
    configuration dictionaries
    """
    algo_config = (
      self.base_algo_config
      # environment
      .environment(environment, env_config=env_config)
    )
    # process the parameters dictionaries
    if ray_config is not None or exp_config is not None:
      env_config["ENV_NAME"] = environment.__name__
      all_params = self.process_config_dictionaries(
        ray_config, exp_config, env_config
      )
      # update the algorithm config
      algo_config.update_from_dict(all_params)
    return algo_config
  
  def process_config_dictionaries(
      self, ray_config: dict, exp_config: dict, env_config: dict
    ) -> dict:
    """
    Processes the configuration dictionaries, extracting the relevant 
    information to define an `AlgorithmConfig`
    """
    all_params = {}
    # merge sub-dictionaries of ray_config
    if ray_config is not None:
      for key, value in ray_config.items():
        if isinstance(value, dict):
          all_params.update(value)
        else:
          all_params.update({key: value})
    # manage "special" keys
    self.update_special_keys(all_params, exp_config, env_config)
    return all_params
  
  def update_special_keys(
      self, all_params: dict, exp_config: dict, env_config: dict
    ):
    """
    Updates the work-in-progress dictionary of parameters by converting the 
    provided keys if necessary
    """
    # check the presence of protected/suggested keys
    using_suggested_keys, using_protected_keys = self.validate_key_usage(
      all_params
    )
    # fix the value of keys that should not be overridden (unless they may 
    # have been manually specified)
    if not using_protected_keys:
      all_params["min_time_s_per_iteration"] = 0
      all_params["min_sample_timesteps_per_iteration"] = 0
      all_params["min_train_timesteps_per_iteration"] = 0
    else:
      print(
        """
        WARNING: 
        `min_*_per_iteration` variables are not forced to 0
        Set them manually or check the default
        """
      )
    # if suggested keys are provided, these should be converted into the 
    # appropriate standard keys
    if using_suggested_keys:
      self.convert_rollout_parameters(all_params, env_config)
      self.convert_resources_parameters(all_params)
      self.convert_training_parameters(all_params)
    # manage the debugging configuration, creating the experiment logdir 
    # if required
    exp_logdir = self.generate_logdir(exp_config, env_config)
    if exp_logdir is not None:
      if "logger_config" not in all_params or all_params["logger_config"] is None:
        all_params["logger_config"] = {}
      all_params["logger_config"]["logdir"] = exp_logdir
  
  def validate_key_usage(self, all_params: dict):
    """
    Checks if the user is setting any protected/suggested key and throws 
    appropriate errors/warnings
    """
    # check if the user is setting any suggested key
    using_suggested_keys = any(k in all_params for k,_ in self._suggested_keys)
    # check if the user is setting any protected key
    using_protected_keys = False
    for pk,_ in self._protected_keys:
      if pk in all_params:
        using_protected_keys = True
        if pk != "logger_config":
          # prevent the user from simultaneously setting protected and 
          # suggested keys
          if using_suggested_keys:
            raise KeyError(
              "ERROR: mixing protected and suggested keys is forbidden"
            )
          # raise a warning otherwise
          else:
            pv = all_params[pk]
            print(
              f"WARNING: manually setting protected key {pk} with value {pv}"
            )
        else:
          # prevent the user from manually setting the logging directory
          if "logdir" in all_params[pk]:
            raise KeyError(
              "ERROR: set a general logging directory from `exp_config.json`"
            )
    return using_suggested_keys, using_protected_keys
  
  def convert_rollout_parameters(self, all_params: dict, env_config: dict):
    """
    Defines the appropriate parameters related to the definition and 
    behavior of rollout workers, according to the provided keys
    """
    # duration unit
    unit = self.base_algo_config["batch_mode"]
    if "duration_unit" in all_params:
      unit = all_params.pop("duration_unit")
      if unit == "step":
        all_params["batch_mode"] = "truncated_episodes"
        unit = "truncated_episodes"
      elif unit == "episode":
        all_params["batch_mode"] = "complete_episodes"
        unit = "complete_episodes"
      else:
        raise ValueError(f"ERROR: invalid `duration_unit` {unit}")
    # duration
    if "duration_per_worker" in all_params:
      duration = all_params.pop("duration_per_worker")
      if unit == "truncated_episodes":
        all_params["rollout_fragment_length"] = duration
      elif unit == "complete_episodes":
        min_time = env_config["min_time"]
        max_time = env_config["max_time"]
        time_step = env_config["time_step"]
        n_steps = (max_time - min_time) // time_step
        all_params["rollout_fragment_length"] = duration * n_steps

  def convert_resources_parameters(self, all_params):
    """
    Defines the appropriate parameters related to the resources requirement, 
    according to the provided keys
    """
    # master CPUs
    if "num_cpus_master" in all_params:
      num_cpus = all_params.pop("num_cpus_master")
      all_params["num_cpus_per_local_worker"] = num_cpus
    # master GPUs
    if "num_gpus_master" in all_params:
      num_gpus = all_params.pop("num_gpus_master")
      all_params["num_gpus"] = num_gpus
  
  @abstractmethod
  def convert_training_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to the definition and 
    behavior of the policy training algorithm, according to the provided keys
    """
    pass

  def generate_logdir(self, exp_config: dict, env_config: dict) -> str:
    """
    Generate the experiment `logdir` if an appropriate parameter is provided
    """
    exp_logdir = None
    if exp_config is not None and "logdir" in exp_config:
      now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
      exp_logdir = os.path.join(
        os.path.abspath(exp_config["logdir"]), 
        f"{self.algo}_{env_config['ENV_NAME']}_{now}"
      )
      os.makedirs(exp_logdir, exist_ok=True)
    return exp_logdir
  
  def to_dict(self, algo_config: AlgorithmConfig) -> dict:
    """
    Converts the given `AlgorithmConfig` into a dictionary
    """
    all_params = algo_config.serialize()
    # split according to the dictionary of class method parameters
    ray_config = {}
    for method, method_params in self.algo_methods.items():
      for param in method_params:
        if param in all_params:
          if method not in ray_config:
            ray_config[method] = {}
          ray_config[method][param] = all_params.pop(param)
    # add those that could not be classified
    ray_config["not_classified"] = {}
    for param, value in all_params.items():
      key = self.base_algo_config._translate_special_keys(param, False)
      added = False
      for method, method_params in self.algo_methods.items():
        if key in method_params:
          ray_config[method][key] = value
          added = True
      if not added:
        ray_config["not_classified"][param] = value
    return ray_config
  
  def to_json(self, algo_config: AlgorithmConfig) -> str:
    """
    Converts the given `AlgorithmConfig` into a string with json format
    """
    algo_dict = self.to_dict(algo_config)
    return json.dumps(algo_dict, indent = 2)

##############################################################################
# PPO
##############################################################################
class PPOConfigGenerator(AlgoConfigGenerator):
  def __init__(self):
    super().__init__()
    self.algo = "PPO"
    self.generate_default_config()
    # algorithm-specific protected/suggested keys
    self._protected_keys += [
      ("sgd_minibatch_size", "training"),
      ("num_sgd_iter", "training")
    ]
    # update the dictionary of algo methods and corresponding parameters
    to_inspect = [
      super(type(self.base_algo_config), self.base_algo_config),
      self.base_algo_config
    ]
    for f_name in dir(self.base_algo_config):
      for elem in to_inspect:
        try:
          f = getattr(elem, f_name)
          if callable(f) and not f_name.startswith("_"):
            f_info = inspect.getfullargspec(f)
            info_set = set(f_info.args + f_info.kwonlyargs)
            if f_name not in self.algo_methods:
              self.algo_methods[f_name] = info_set
            else:
              self.algo_methods[f_name] = self.algo_methods[f_name].union(
                info_set
              )
        except Exception as e:
          self.algo_methods[f_name] = {str(e)}

  def convert_training_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to the definition and 
    behavior of the policy training algorithm, according to the provided keys
    """
    # train batch size
    n_steps = self.base_algo_config["train_batch_size"]
    if "rollout_fragment_length" in all_params:
      nw = all_params.get(
        "num_rollout_workers",
        max(self.base_algo_config["num_rollout_workers"], 1)
      )
      n_steps = nw * all_params["rollout_fragment_length"]
      all_params["train_batch_size"] = n_steps
      print(
        f"INFO: {nw} rollout workers will collect overall {n_steps} steps"
      )
    # sgd batch size
    if "batch_size" in all_params:
      batch_size = all_params.pop("batch_size")
      all_params["sgd_minibatch_size"] = batch_size
      print(
        f"INFO: training batches will have size: {batch_size}"
      )
    # number of sgd iterations
    if "num_trained_batches" in all_params:
      num_batches = all_params.pop("num_trained_batches")
      all_params["num_sgd_iter"] = num_batches
      print(
       f"INFO: {num_batches} batches will be extracted for training"
      )

##############################################################################
# DQN
##############################################################################
class DQNConfigGenerator(AlgoConfigGenerator):
  def __init__(self):
    super().__init__()
    self.algo = "DQN"
    self.generate_default_config()
    # algorithm-specific protected/suggested keys
    self._protected_keys += [
      ("training_intensity", "training")
    ]
    # update the dictionary of algo methods and corresponding parameters
    to_inspect = [
      super(type(self.base_algo_config), self.base_algo_config),
      self.base_algo_config
    ]
    for f_name in dir(self.base_algo_config):
      for elem in to_inspect:
        try:
          f = getattr(elem, f_name)
          if callable(f) and not f_name.startswith("_"):
            f_info = inspect.getfullargspec(f)
            info_set = set(f_info.args + f_info.kwonlyargs)
            if f_name not in self.algo_methods:
              self.algo_methods[f_name] = info_set
            else:
              self.algo_methods[f_name] = self.algo_methods[f_name].union(
                info_set
              )
        except Exception as e:
          self.algo_methods[f_name] = {str(e)}
  
  def convert_training_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to the definition and 
    behavior of the policy training algorithm, according to the provided keys
    """
    # train batch size
    batch_size = self.base_algo_config["train_batch_size"]
    if "batch_size" in all_params:
      batch_size = all_params.pop("batch_size")
      all_params["train_batch_size"] = batch_size
      print(
        f"INFO: training batches will have size {batch_size}"
      )
    # training intensity
    if "num_trained_batches" in all_params:
      num_batches = all_params.pop("num_trained_batches")
      # number of sampled steps
      nw = all_params.get(
        "num_rollout_workers",
        max(1, self.base_algo_config["num_rollout_workers"])
      )
      rfl = all_params.get(
        "rollout_fragment_length",
        self.base_algo_config["rollout_fragment_length"]
      )
      n_sampled_steps = nw * rfl
      print(
        f"INFO: {nw} workers will collect overall {n_sampled_steps} steps"
      )
      # number of trained steps & intensity
      if num_batches > 1:
        n_trained_steps = batch_size * num_batches
        all_params["training_intensity"] = n_trained_steps // n_sampled_steps
        print(
          f"INFO: total number of trained steps is {n_trained_steps}"
        )
      else:
        all_params["training_intensity"] = None
