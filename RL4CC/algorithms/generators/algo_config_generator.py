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
from RL4CC.utilities.common import not_defined
from RL4CC.utilities.logger import Logger

from ray.rllib.algorithms import AlgorithmConfig
from ray.tune.registry import get_trainable_cls
from ray.tune.search.sample import Domain
from abc import ABC, abstractmethod
from collections import namedtuple
from ray import tune
import numpy as np
import inspect
import json


class AlgoConfigGenerator(ABC):
  ParameterDomain = namedtuple("ParameterDomain", "value lower upper")
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ):
    self.logger = logger
    self.algo = None
    self.base_algo_config = None
    self.algo_methods = None
    self.use_tune = False
    self._protected_keys = [
      # (key, key group)
      ("rollout_fragment_length", "rollouts"),
      ("batch_mode", "rollouts"),
      ("train_batch_size", "training"),
      ("min_time_s_per_iteration", "reporting"),
      ("min_sample_timesteps_per_iteration", "reporting"),
      ("min_train_timesteps_per_iteration", "reporting"),
      ("num_gpus", "resources"),
      ("num_cpus_for_local_worker", "resources"),
      ("logger_config", "debugging"),
      ("evaluation_interval", "evaluation"),
      ("evaluation_duration", "evaluation"),
      ("env", "environment"),
      ("env_config", "environment")
    ]
    self._suggested_keys = [
      # (key, key group, tunable)
      ("duration_per_worker",  "rollouts", "tunable"),
      ("duration_unit",  "rollouts", "not_tunable"),
      ("batch_size",  "training", "tunable"),
      ("num_train_batches",  "training", "tunable"),
      ("num_gpus_master",  "resources", "not_tunable"),
      ("num_cpus_master",  "resources", "not_tunable"),
      ("evaluation_duration_per_worker", "evaluation", "not_tunable")
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
      env_config: dict,
      ray_config: dict = None,
      exp_logdir: str = None,
      eval_interval: int = None,
      use_tune: bool = False,
    ) -> AlgorithmConfig:
    """
    Defines the `AlgorithmConfig` considering the provided environment and
    configuration dictionaries
    """
    if env_config is None:
      raise RuntimeError(
        "ERROR: cannot create an algorithm without environment"
      )
    if "env_name" not in env_config:
      raise KeyError(
        "ERROR: cannot create an environment without a name"
      )
    if use_tune:
      self.use_tune = True
    # start config generation
    algo_config = (
      self.base_algo_config
      # environment
      .environment(
        env_config["env_name"],
        # pass-along config dictionary avoiding env_name
        env_config={k:v for k,v in env_config.items() if k != "env_name"}
      )
    )
    # process the configuration parameters
    all_params = self.process_config_parameters(
      ray_config, env_config, exp_logdir, eval_interval
    )
    # update the algorithm config
    if len(all_params) > 0:
      algo_config.update_from_dict(all_params)
      # properly set the environment configuration for validation (if required)
      eval_config = all_params.pop("evaluation_config", {})
      if len(eval_config) > 0:
        algo_config.evaluation(
          evaluation_config = AlgorithmConfig.overrides(
            env_config = eval_config
          )
        )
    # validate the number of collected and trained steps
    self.validate_collection_and_training_size(algo_config)
    return algo_config
  
  def generate_eval_config(
      self, env_config: dict, new_env_config: dict
    ) -> dict:
    """
    Generate and evaluation environment config by suitably modifying the 
    original one through the given parameters
    """
    eval_config = {**env_config}
    for key, val in new_env_config.items():
      eval_config[key] = val
    return eval_config

  def process_config_parameters(
      self,
      ray_config: dict,
      env_config: dict,
      exp_logdir: str = None,
      eval_interval: int = None
    ) -> dict:
    """
    Processes the configuration parameters, extracting the relevant
    information to define an `AlgorithmConfig`
    """
    all_params = {}
    # merge sub-dictionaries of ray_config
    if ray_config is not None:
      for key, value in ray_config.items():
        # check the existence of Tuning Strings, convert them to tune objects 
        # wherever they exist
        value = self.interpret_tune_config(key, value)
        if isinstance(value, dict):
          all_params.update(value)
        else:
          all_params.update({key: value})
    # manage "special" keys
    self.update_special_keys(
      all_params, env_config, exp_logdir, eval_interval
    )
    return all_params

  def update_special_keys(
      self,
      all_params: dict,
      env_config: dict,
      exp_logdir: str = None,
      eval_interval: int = None
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
      all_params["metrics_num_episodes_for_smoothing"] = 1
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
    # process the evaluation interval
    self.convert_evaluation_parameters(all_params, env_config, eval_interval)
    # manage the debugging configuration, creating the experiment logdir 
    # if required
    if exp_logdir is not None:
      if not_defined("logger_config", all_params):
        all_params["logger_config"] = {}
      if not_defined("type", all_params["logger_config"]):
        all_params["logger_config"]["type"] = "ray.tune.logger.UnifiedLogger"
      all_params["logger_config"]["logdir"] = exp_logdir
    else:
      all_params["logger_config"] = None
  
  def validate_key_usage(self, all_params: dict):
    """
    Checks if the user is setting any protected/suggested key and throws
    appropriate errors/warnings
    """
    # check if the user is setting any suggested key
    using_suggested_keys = any(
      k in all_params for k,_ , _ in self._suggested_keys
    )
    # check if the user is setting any protected key
    using_protected_keys = False
    for pk,_ in self._protected_keys:
      if pk in all_params:
        # prevent the user from improperly setting the environment config
        if pk == "env" or pk == "env_config":
          raise KeyError(
            "ERROR: `env` and `env_config` cannot be manually configured"
          )
        # prevent the user from improperly setting the evaluation interval
        elif pk == "evaluation_interval":
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
          using_protected_keys = True
          # prevent the user from simultaneously setting protected and
          # suggested keys
          if using_suggested_keys:
            msg = "ERROR: mixing protected and suggested keys is forbidden"
            raise KeyError(
              msg + f" (protected key: `{pk}`)"
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
        all_params["batch_mode"] = "truncate_episodes"
        unit = "truncate_episodes"
      elif unit == "episodes":
        all_params["batch_mode"] = "complete_episodes"
        unit = "complete_episodes"
      else:
        raise ValueError(f"ERROR: invalid `duration_unit` {unit}")
    # duration
    if "duration_per_worker" in all_params:
      duration = all_params.pop("duration_per_worker")
      if unit == "truncate_episodes":
        all_params["rollout_fragment_length"] = duration
      elif unit == "complete_episodes":
        n_steps = self.compute_num_steps_per_episode(env_config)
        all_params["rollout_fragment_length"], _, _ = self.scale_parameter(
          duration, scale_factor = n_steps
        )

  def convert_evaluation_parameters(
      self, all_params: dict, env_config: dict, eval_interval: int = None
    ):
    """
    Defines the appropriate parameters related to the evaluation length,
    according to the provided keys
    """
    # evaluation interval
    if eval_interval is not None and not np.isinf(eval_interval):
      all_params["evaluation_interval"] = eval_interval
    # duration unit
    unit = all_params.get(
      "evaluation_duration_unit",
      self.base_algo_config["evaluation_duration_unit"]
    )
    if unit != "timesteps" and unit != "episodes":
      raise ValueError(f"ERROR: invalid `evaluation_duration_unit` {unit}")
    # duration
    num_workers = all_params.get(
      "evaluation_num_workers",
      max(1, self.base_algo_config["evaluation_num_workers"])
    )
    duration = all_params.get(
      "evaluation_duration",  # needed if the method is called when 
                              # using protected keys
      self.base_algo_config["evaluation_duration"]
    )
    if "evaluation_duration_per_worker" in all_params:
      duration = all_params.pop("evaluation_duration_per_worker") * num_workers
      all_params["evaluation_duration"] = duration
    # guarantee that at least the final evaluation can be surely performed
    # (force the local (non-eval) worker to have an environment to evaluate on)
    if not_defined("evaluation_interval", all_params):
      all_params["create_env_on_driver"] = True
      self.logger.warn(
        "no `evaluation_interval` is set in `exp_config.json`. "
        "A final evaluation will still be performed, with "
        f"{num_workers} worker(s) collecting overall {duration} {unit}"
      )
    # define the environment parameters for algorithm evaluation
    if "evaluation_config" in all_params:
      all_params["evaluation_config"] = self.generate_eval_config(
        env_config, all_params["evaluation_config"]
      )

  def convert_resources_parameters(self, all_params):
    """
    Defines the appropriate parameters related to the resources requirement,
    according to the provided keys
    """
    # master CPUs
    if "num_cpus_master" in all_params:
      num_cpus = all_params.pop("num_cpus_master")
      all_params["num_cpus_for_local_worker"] = num_cpus
    # master GPUs
    if "num_gpus_master" in all_params:
      num_gpus = all_params.pop("num_gpus_master")
      all_params["num_gpus"] = num_gpus

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
    self.logger.breakline(1)
    self.logger.log(
      f"*** sampled/trained steps in each `{self.algo}.training_step()` ***", 1
    )
    tot_sampled = self.count_sampled_steps(algo_config)
    tot_trained = self.count_trained_steps(algo_config)
    self.logger.breakline(1)
    # raise a WARNING if the number of collected and trained steps are too 
    # unbalanced
    if tot_trained.lower < tot_sampled.lower * 0.9:
      self.logger.warn(
        f"only {self.replace_tune_objects(tot_trained.value)} trained steps "
        f"over the {self.replace_tune_objects(tot_sampled.value)} sampled"
      )
    elif tot_trained.upper > tot_sampled.upper * 1.1:
      self.logger.warn(
        f"{self.replace_tune_objects(tot_trained.value)} steps trained over "
        f"only {self.replace_tune_objects(tot_sampled.value)} new samples"
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
  def count_sampled_steps(
      self, algo_config: AlgorithmConfig
    ) -> ParameterDomain:
    """
    Counts the number of sampled steps according to the given `AlgorithmConfig`
    """
    pass

  @abstractmethod
  def count_trained_steps(
      self, algo_config: AlgorithmConfig
    ) -> ParameterDomain:
    """
    Counts the number of trained steps according to the given `AlgorithmConfig`
    """
    pass

  def to_dict(self, algo_config: AlgorithmConfig) -> dict:
    """
    Converts the given `AlgorithmConfig` into a dictionary
    """
    try:
      all_params = algo_config.serialize()
    except:
      all_params = AlgorithmConfig._serialize_dict(algo_config)
      pass
    # intermediate step in case of tune.search.Domain (when tuning) exist
    # the method will replace the tune objetcs with a string to make it json 
    # serializable
    all_params = self.replace_tune_objects(all_params)
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
    Converts the given `AlgorithmConfig` or config dictionary into a string 
    with json format
    """
    algo_dict = self.to_dict(algo_config)
    return json.dumps(algo_dict, indent = 2)

  def replace_tune_objects(self, config):
    """
    Replaces Ray Tune objects in a dictionary with their string repr
    """
    if isinstance(config, dict):
      return {k: self.replace_tune_objects(v) for k, v in config.items()}
    elif isinstance(config, list):
      return [self.replace_tune_objects(v) for v in config]
    elif self.is_tuned(config):
      # Return the string representation of the tune object
      domain = config.domain_str
      sampler = config.sampler.__str__().lower()
      tune_string = f"tune.{sampler}{domain}"
      return tune_string
    else:
      return config

  def interpret_tune_config(self, config_key, config_value):
    """
    Interprets a configuration value, converting it to a format usable by 
    ray.tune.
    Handles dictionaries, lists within dictionaries, nested lists, and 
    strings representing Ray Tune objects.

    :param config_key: The configuration key.
    :param config_value: The configuration value to interpret.
    :return: The interpreted value.
    """
    try:
      # dictionaries
      if isinstance(config_value, dict):
        return {
          key: self.interpret_tune_config(
            key, value
          ) for key, value in config_value.items()
        }
      # lists
      if isinstance(config_value, list):
        return [
          self.interpret_tune_config(
            config_value, item
          ) for item in config_value
        ]
      # strings
      if isinstance(config_value, str):
        if config_value.strip().startswith("tune."):
          # check whether tune is enabled
          if self.use_tune:
            self.validate_special_key_tuning(config_key = config_key)
            self.logger.log(f'Tuning detected for the value of "{config_key}"')
            return eval(config_value.strip())
          else:
            raise ValueError(
              "Hyperparameter tuning is not supported in single experiments "
              "remove tune.search_space objects in the algorithm config files."
            )
        else:
          return config_value
      else:
        return config_value
    except Exception as e:
      raise ValueError(f"Error interpreting algorithm configuration: {e}")

  def validate_special_key_tuning(self, config_key):
    """
    Validate the given configuration key checking whether it is a tunable 
    parameter
    """
    # define the special keys
    special_keys = [key for key, _, _ in self._suggested_keys]
    # define the special keys that can be tuned
    tunable_special_keys = [
      key for key, _, tunable in self._suggested_keys if tunable == "tunable"
    ]
    # check if the special key is tunable
    if config_key in special_keys:
      if config_key not in tunable_special_keys:
        # interrupts execution if the value is not tunable
        raise ValueError(
          f"The parameter `{config_key}` is a non-tunable parameter"
        )

  def scale_parameter(
      self, config_value, scale_factor = 1, addend = 0
    ) -> ParameterDomain:
    """
    This method is to be mainly used when scaling config parameters.

    It will return the multiplication of the config_value and the scale_factor 
    when they are of a generic data type.

    It will return the scaling of the config value when provided tune.search 
    objects

    Args:
      config_value: The term to be scaled. This can be a Ray Tune object 
        (such as tune.uniform, tune.loguniform, etc.).
      scale_factor: The scaling object. This can be a Ray Tune object.
      addend: The object to add. This can be a Ray Tune objec

    Returns:
        An `AlgoConfigGenerator.ParameterDomain` tuple reporting the value, 
        lower and upper limits of scaled config_value
    """
    scaled_domain = None
    # check if the value is a tune object
    if self.is_tuned(config_value):
      # scale the bounds of the tune objects with the scale factor
      scaled_domain = self.scale_tune_object(
        config_value, scale_factor, addend
      )
    elif self.is_tuned(scale_factor):
      scaled_domain = self.scale_tune_object(
        scale_factor, config_value, addend
      )
    else:
      # Normally scale the parameter
      val = config_value * scale_factor + addend
      scaled_domain = self.ParameterDomain(val, val, val)
    return scaled_domain

  def scale_tune_object(
      self, obj, factor = 1, addend = 0
    ) -> ParameterDomain:
    """
    Scale the bounds or values of a Ray Tune object by a factor.

    Args:
      obj: The term to be scaled. This can be a Ray Tune object 
        (such as tune.uniform, tune.loguniform, etc.).
      factor: The scaling object. This can be a Ray Tune object.
      addend: The object to add. This can be a Ray Tune object.

    Returns:
        A new Ray Tune object with scaled bounds or values.
    """
    # first, identify the sampler type of the object
    obj_type = self.identify_sampler_type(obj)
    # check if the factor or the addend are ray objects; interrupt execution 
    # when samplers are mismatched. When they are Ray tune objects, the upper 
    # and lower bounds are fetched; otherwise, the same factor is assigned to 
    # both
    if self.is_tuned(factor):
      upper_factor, lower_factor = self.check_sampler_compatibility(
        obj = obj, 
        operand = factor
      )
    else:
      upper_factor = lower_factor = factor
    # addend
    if self.is_tuned(addend):
      upper_addend, lower_addend = self.check_sampler_compatibility(
        obj = obj, 
        operand = addend
      )
    else:
      upper_addend = lower_addend = addend
    # distribution
    if obj_type in {
        "uniform", 
        "loguniform", 
        "quniform", 
        "qloguniform", 
        "randint", 
        "qrandint"
      }:
      lower = obj.lower * lower_factor + upper_addend
      upper = obj.upper * upper_factor + lower_addend
      if obj_type == "uniform":
        return self.ParameterDomain(
          tune.uniform(lower, upper), lower, upper
        )
      elif obj_type == "loguniform":
        return self.ParameterDomain(
          tune.loguniform(lower, upper), lower, upper
        )
      elif obj_type == "quniform":
        return self.ParameterDomain(
          tune.quniform(lower, upper, obj.q), lower, upper
        )
      elif obj_type == "qloguniform":
        return self.ParameterDomain(
          tune.qloguniform(lower, upper, obj.q), lower, upper
        )
      if obj_type == "randint":
        return self.ParameterDomain(
          tune.randint(lower, upper), lower, upper
        )
      elif obj_type == "qrandint":
        return self.ParameterDomain(
          tune.qrandint(lower, upper, obj.sampler.q), lower, upper
        )
      else:
        raise ValueError(f"Unsupported Ray Tune object: {obj}")
    elif obj_type == "choice":
      values_list = [value * factor + addend for value in obj.categories]
      return self.ParameterDomain(
        tune.grid_search(values_list), min(values_list), max(values_list)
      )
    else:
      raise ValueError(f"Unsupported Ray Tune object: {obj}")

  @staticmethod
  def compute_num_steps_per_episode(env_config: dict) -> int:
    """
    Compute the number of steps per episode based on the environment 
    configuration
    """
    n_steps = None
    if all(k in env_config for k in ["min_time", "max_time", "time_step"]):
      min_time = env_config["min_time"]
      max_time = env_config["max_time"]
      time_step = env_config["time_step"]
      n_steps = (max_time - min_time) // time_step
    else:
      raise ValueError(
        "ERROR: not enough parameters to support `episodes` duration. "
        "Check if env_config.json includes `min_time`, `max_time`, "
        "`time_step`"
      )
    return n_steps
  
  @staticmethod
  def is_tuned(key) -> bool:
    return isinstance(key, Domain)
  
  @staticmethod
  def identify_sampler_type(sampler):
    """
    Identify the type of Ray Tune search sampler
    """
    if isinstance(sampler, tune.search.sample.Quantized):
      base_type = AlgoConfigGenerator.identify_sampler_type(
        sampler.base_sampler
      )
      return f"q{base_type}"
    elif isinstance(sampler, tune.search.sample.Uniform):
      return "uniform"
    elif isinstance(sampler, tune.search.sample.LogUniform):
      return "loguniform"
    elif isinstance(sampler, tune.search.sample.Integer):
      if isinstance(sampler.sampler, tune.search.sample.Uniform):
        return "randint"
      elif isinstance(sampler.sampler, tune.search.sample.LogUniform):
        return "lograndint"
      elif isinstance(sampler.sampler, tune.search.sample.Quantized):
        return "qrandint"
    elif isinstance(sampler, tune.search.sample.Categorical):
      return "choice"
    else:
      return "unknown"

  def check_sampler_compatibility(self, obj, operand):
    obj_sampler_type = self.identify_sampler_type(obj)
    operand_sampler_type = self.identify_sampler_type(operand)
    if obj_sampler_type != operand_sampler_type:
      raise NotImplementedError(
        f"The samplers of type {obj_sampler_type} and {operand_sampler_type} "
        "are mismatched."
      )
    else:
      if operand_sampler_type in {
          "uniform", 
          "loguniform", 
          "quniform", 
          "qloguniform", 
          "randint", 
          "qrandint"
        }:
        upper_bound = obj.lower
        lower_bound = obj.upper
        return upper_bound, lower_bound
      elif operand_sampler_type == "choice":
        raise NotImplementedError(
          f"The samplers of type {obj_sampler_type} and {operand_sampler_type} "
          "are mismatched."
        )