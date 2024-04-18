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
from environment.base_environment import BaseEnvironment
from utilities.common import not_defined
from utilities.logger import Logger

from ray.rllib.algorithms import AlgorithmConfig
from ray.tune.registry import get_trainable_cls
from abc import ABC, abstractmethod
from datetime import datetime
import inspect
import json
import os


class AlgoConfigGenerator(ABC):
  def __init__(self):
    self.logger = Logger(name="RL4CC-AlgoConfigGenerator")
    self.algo = None
    self.base_algo_config = None
    self.algo_methods = None
    self._protected_keys = [
      # (key, key group)
      ("rollout_fragment_length", "rollouts"),
      ("batch_mode", "rollouts"),
      ("train_batch_size", "training"),
      ("min_time_s_per_iteration", "reporting"),
      ("min_sample_timesteps_per_iteration", "reporting"),
      ("min_train_timesteps_per_iteration", "reporting"),
      ("num_gpus", "resources"),
      ("num_cpus_per_local_worker", "resources"),
      ("evaluation_interval", "evaluation")
    ]
    self._suggested_keys = [
      # (key, key group)
      ("duration_per_worker",  "rollouts"),
      ("duration_unit",  "rollouts"),
      ("batch_size",  "training"),
      ("num_train_batches",  "training"),
      ("num_gpus_master",  "resources"),
      ("num_cpus_master",  "resources")
    ]
  
  def save_algo_methods_dict(self):
    """
    Saves a dictionary of algo methods and corresponding parameters (useful 
    to print the `AlgorithmConfig`)
    """
    self.algo_methods = {}
    # to get the complete list of methods and corresponding parameters, we 
    # have to inspect all parents of the `{self.algo}Config` class up to 
    # `AlgorithmConfig`
    all_parents = inspect.getmro(self.base_algo_config.__class__)
    found_AlgorithmConfig = False
    idx = 0
    while not found_AlgorithmConfig:
      class_to_inspect = all_parents[idx]
      # loop over all members/methods of the current class
      for f_name in dir(class_to_inspect):
        try:
          # if the current element is a public method...
          f = getattr(class_to_inspect, f_name)
          if callable(f) and not f_name.startswith("_"):
            # ...save the list of arguments and keyword arguments
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
      # check if we have reached the last parent to inspect
      found_AlgorithmConfig = (class_to_inspect.__name__ == "AlgorithmConfig")
      idx += 1
  
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
    # validate the number of collected and trained steps
    self.validate_collection_and_training_size(algo_config)
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
      msg = "`min_*_per_iteration` variables are not forced to 0. "
      self.logger.warn(
        msg + "Set them manually or check the default"
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
      if not_defined("logger_config", all_params):
        all_params["logger_config"] = {}
      all_params["logger_config"]["logdir"] = exp_logdir
    # manage the evaluation configuration, so that at least the final 
    # evaluation can be surely performed (force the local (non-eval) worker 
    # to have an environment to evaluate on)
    if exp_config is not None and "evaluation_interval" in exp_config:
      all_params["evaluation_interval"] = exp_config["evaluation_interval"]
    if not_defined("evaluation_interval", all_params):
      all_params["create_env_on_driver"] = True
  
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
        # prevent the user from improperly setting the evaluation interval
        if pk == "evaluation_interval":
          raise KeyError(
            "ERROR: set the evaluation interval from `exp_config.json`"
          )
        # prevent the user from manually setting the logging directory
        elif pk == "logger_config":
          if "logdir" in all_params[pk]:
            raise KeyError(
              "ERROR: set a general logging directory from `exp_config.json`"
            )
        else:
          # prevent the user from simultaneously setting protected and 
          # suggested keys
          if using_suggested_keys:
            raise KeyError(
              "ERROR: mixing protected and suggested keys is forbidden"
            )
          # raise a warning otherwise
          else:
            pv = all_params[pk]
            self.logger.warn(
              f"manually setting protected key `{pk}` with value: {pv}"
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
      if unit == "timesteps":
        all_params["batch_mode"] = "truncated_episodes"
        unit = "truncated_episodes"
      elif unit == "episodes":
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
  
  def check_num_training_step_calls(self, algo_config: AlgorithmConfig):
    """
    Checks if the `training_step` function will be called more than once 
    according to the given `AlgorithmConfig`
    """
    mins = algo_config["min_sample_timesteps_per_iteration"]
    mint = algo_config["min_train_timesteps_per_iteration"]
    if mins > 0 or mint > 0:
      msg = f"To collect at least {mins} step(s) and train at least {mint} "
      msg += f"step(s) in each call to `{self.algo}.train()`, the "
      msg += f"`{self.algo}.training_step()` method may be executed "
      self.logger.warn(
        msg + "more than once"
      )
  
  def validate_collection_and_training_size(
      self, algo_config: AlgorithmConfig
    ):
    """
    Computes the number of sampled and trained steps according to the 
    given `AlgorithmConfig` and checks whether the values are coherent
    """
    self.logger.breakline()
    self.logger.log(
      f"*** sampled/trained steps in each `{self.algo}.training_step()` ***"
    )
    tot_sampled = self.count_sampled_steps(algo_config)
    tot_trained = self.count_trained_steps(algo_config)
    self.logger.breakline()
    # raise a WARNING if the number of collected and trained steps are too 
    # unbalanced
    if tot_trained < tot_sampled * 0.9:
      self.logger.warn(
        f"only {tot_trained} steps are trained over the {tot_sampled} sampled"
      )
    elif tot_trained > tot_sampled * 1.1:
      self.logger.warn(
        f"{tot_trained} steps trained over only {tot_sampled} new samples"
      )
    # check if the `training_step` function will be called more than once
    self.check_num_training_step_calls(algo_config)
  
  @abstractmethod
  def convert_training_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to the definition and 
    behavior of the policy training algorithm, according to the provided keys
    """
    pass

  @abstractmethod
  def count_sampled_steps(self, algo_config: AlgorithmConfig) -> int:
    """
    Counts the number of sampled steps according to the given `AlgorithmConfig`
    """
    pass

  @abstractmethod
  def count_trained_steps(self, algo_config: AlgorithmConfig) -> int:
    """
    Counts the number of trained steps according to the given `AlgorithmConfig`
    """
    pass
  
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
          value = all_params.pop(param)
          if param == "rl_module_spec":
            value = str(value.__class__)
          ray_config[method][param] = value
    # add those that could not be classified
    ray_config["not_classified"] = {}
    for param, value in all_params.items():
      key = self.base_algo_config._translate_special_keys(param, False)
      added = False
      for method, method_params in self.algo_methods.items():
        if key in method_params:
          if method not in ray_config:
            ray_config[method] = {}
          if key == "rl_module_spec":
            value = str(value.__class__)
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
