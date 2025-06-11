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
from RL4CC.utilities.logger import Logger

from ray.rllib.utils.serialization import deserialize_type
from ray.tune.search.hyperopt import HyperOptSearch
from ray.air import RunConfig, CheckpointConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune import TuneConfig
from datetime import datetime
from typing import Tuple
import os


class TuneConfigGenerator:
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-TuneConfigGenerator")
    ):
    self.logger = logger
    self._required_keys = ["num_tune_trials", "metric", "mode"]

  def generate_tune_config(self, tune_config: dict) -> TuneConfig:
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
    tune_params = TuneConfig(
      **tune_config_dict,
      trial_name_creator = self.trial_name_string,
      trial_dirname_creator = self.trial_name_string
    )
    return tune_params
  
  def generate_checkpoint_config(
      self, checkpoint_config: dict
    ) -> CheckpointConfig:
    """
    Generates a `CheckpointConfig` object based on the provided configuration 
    parameters
    """
    config = None
    if checkpoint_config is not None:
      config = CheckpointConfig(**checkpoint_config)
    return config

  def generate_run_config(
      self,
      stopping_criterion: dict,
      storage_path: str = None,
      callbacks: list = None,
      checkpoint_config: CheckpointConfig = None,
      progress_reporter: dict = None
    ) -> RunConfig:
    """
    Generates a `RunConfig` object based on the provided configuration 
    parameters
    """
    name = None
    progress_file = None
    if storage_path is not None:
      now = datetime.now().strftime('%H-%M-%S.%f')
      name = f"tuning_experiment_output_{now}"
      progress_file = os.path.join(storage_path, "exp_progress.json")
    # define progress reporter
    progress_reporter_param = {}
    if progress_reporter is not None:
      progress_reporter_obj = deserialize_type(
        progress_reporter["progress_reporter_class"]
      )
      progress_reporter_config = progress_reporter.get(
        "progress_reporter_config", {}
      )
      # add progress file path if required but defined as None
      if "progress_file" in progress_reporter_config:
        if progress_reporter_config["progress_file"] is None:
          progress_reporter_config["progress_file"] = progress_file
      # update parameter
      progress_reporter_param = {
        "progress_reporter": progress_reporter_obj(**progress_reporter_config)
      }
    # generate RunConfig
    run_config = RunConfig(
      name = name,
      verbose = self.convert_verbosity_level(),
      stop = stopping_criterion,
      storage_path = storage_path,
      callbacks = callbacks,
      checkpoint_config = checkpoint_config,
      **progress_reporter_param
    )
    return run_config
  
  def generate(
      self,
      tune_config: dict,
      stopping_criterion: dict,
      storage_path: str = None,
      callbacks: list = None,
      checkpoint_config: dict = None
    ) -> Tuple[TuneConfig, RunConfig]:
    """
    Generates the `TuneConfig` and `RunConfig` objects based on the provided 
    configuration parameters
    """
    # extract the progress reporter configuration from tune_config (if 
    # provided)
    progress_reporter = tune_config.pop("progress_reporter", None)
    # generate `TuneConfig`
    tune = self.generate_tune_config(tune_config)
    # generate `CheckpointConfig`
    checkpoint = self.generate_checkpoint_config(checkpoint_config)
    # load callbacks
    # generate `RunConfig`
    run = self.generate_run_config(
      stopping_criterion = stopping_criterion,
      storage_path = storage_path,
      callbacks = callbacks,
      checkpoint_config = checkpoint,
      progress_reporter = progress_reporter
    )
    return tune, run

  def validate_tune_config(self, tune_config: dict):
    """
    Validate the configuration dictionary checking for the existence of the 
    mandatory keys
    """
    if any(key not in tune_config for key in self._required_keys):
      raise KeyError(
        f"One or more of the mandatory keys {self._required_keys} "
        "are missing from the tune_config file"
      )
  
  def convert_verbosity_level(self) -> int:
    """
    0 = silent, 1 = default, 2 = verbose
    """
    v = 2
    if self.logger.verbose == 0:
      v = 0
    elif 1 <= self.logger.verbose <= 2:
      v = 1
    return v
  
  @staticmethod
  def trial_name_string(trial):
    """
    Create a custom name for the trial
    """
    return f"{trial.trainable_name}_{trial.trial_id}"

