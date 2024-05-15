"""
Copyright 2024 Mohanad Diab, Federica Filippini

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

from utilities.logger import Logger

from ray import tune, air
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch


class TuneConfigGenerator:
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-TuneConfigGenerator")
    ):
    self.logger = logger
    self._required_keys = ["num_tune_trials", "metric", "mode"]

  def get_tune_config(self, tune_config: dict) -> tune.TuneConfig:
    """
    Generates a `TuneConfig` object based on the provided configuration 
    dictionary
    """
    # get a copy of the tune_config
    tune_config_dict = tune_config.copy()
    self.validate_tune_config(tune_config_dict)
    # handle keys to pass as a parse the tuning dictionary as keyword arguments
    tune_config_dict["num_samples"] = tune_config_dict.pop("num_tune_trials")
    # convert the search algorithm to the respective tune objects
    if "search_algorithm" in tune_config_dict:
      search_algorithm = list(
        tune_config_dict.get("search_algorithm").keys()
      )[0]
      search_algorithm_config = tune_config_dict.get(
        "search_algorithm"
      ).get(search_algorithm)
      if search_algorithm == "hyperopt_search":
        try:
          tune_config_dict["search_alg"] = HyperOptSearch(
            **search_algorithm_config
          )
          tune_config_dict.pop("search_algorithm")
        except Exception:
          raise KeyError(
            "Parameters passed to the hyperopt search algorithm are invalid!"
          )
      else:
        raise NotImplementedError(
          f"Search algorithm {search_algorithm} is not supported"
        )
    # convert the scheduler to the respective tune objects
    if "scheduler" in tune_config_dict:
      scheduler = list(tune_config_dict.get("scheduler").keys())[0]
      scheduler_config = tune_config_dict.get("scheduler").get(scheduler)
      if scheduler == "asha_scheduler":
        try:
          tune_config_dict["scheduler"] = ASHAScheduler(**scheduler_config)
        except:
          raise KeyError(
            "Parameters passed to the ASHAScheduler scheduler are invalid!"
          )
      else:
        raise NotImplementedError(f"Scheduler {scheduler} is not supported")
    # generate `TuneConfig`
    tune_params = tune.TuneConfig(
      **tune_config_dict,
      trial_name_creator = self.trial_name_string,
      trial_dirname_creator = self.trial_name_string
    )
    return tune_params

  def get_run_config(
      self,
      tune_file_name: str = None,
      training_iterations: int = None,
      storage_path: str = None,
      callbacks = None
    ) -> air.RunConfig:
    """
    Generates a `RunConfig` object based on the provided configuration 
    parameters
    """
    run_config = air.RunConfig(
      name = tune_file_name,
      verbose = 1,
      stop = {"training_iteration": training_iterations},
      storage_path = storage_path,
      callbacks = callbacks
    )
    return run_config

  def validate_tune_config(self, tune_config: dict):
    """
    Validate the configuration dictionary checking for the existence of the 
    mandatory keys
    """
    if not all(key in tune_config for key in self._required_keys):
      raise KeyError(
        "One or more of the mandatory keys (num_tune_trials, metric, mode) "
        "are missing from the tune_config file"
      )
  
  @staticmethod
  def trial_name_string(trial):
    """
    Create a custom name for the trial
    """
    return f"{trial.trainable_name}_{trial.trial_id}"

