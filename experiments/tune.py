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
from experiments.base_experiment import BaseExperiment
from algorithms.algorithm import Algorithm
from algorithms.generators.tune_config_generator import TuneConfigGenerator
from utilities.common import not_defined, load_config_file, write_config_file
from utilities.logger import Logger

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray import tune, air
from datetime import datetime
import json
import os


class TuningExperiment(BaseExperiment):
  def __init__(
      self, exp_config_file: str, logger: Logger = Logger(name = "RL4CC")
    ):
    super().__init__(exp_config_file, logger)

  def run(
      self,
      callbacks: DefaultCallbacks = None
    ):
    # the tune configuration must be provided
    if self.tune_config is None:
      raise FileNotFoundError(
        "A tune_config_file must be provided in the experiment configuration"
      )
    else:
      self.tune_config = load_config_file(self.tune_config)
      if self.tune_config is None:
        raise FileNotFoundError(
          "The tune_config_file cannot be accessed"
        )
      use_tune = True
    # get tune params
    tune_config_generator = TuneConfigGenerator()
    tune_config = tune_config_generator.get_tune_config(
      tune_config = self.tune_config
    )
    # define algorithm
    algo = Algorithm(
      algo_name = self.exp_config["algorithm"],
      checkpoint_path = self.checkpoint_path,
      env_config = self.env_config,
      ray_config = self.ray_config,
      base_logdir = self.logdir,
      eval_interval = self.evaluation_interval,
      logger = self.logger,
      use_tune = use_tune
    )
    self.logdir =  algo.logdir
    # save experiment configuration files
    self.write_config_files()
    algo.print_algo_config()
    # prepare Run & Search Space config params
    algo_name = self.exp_config.get("algorithm")
    training_iterations = self.exp_config.get(
      "stopping_criteria", {}
    ).get("max_iterations", 10)
    param_space = algo.algo_config
    try:
      run_config = tune_config_generator.get_run_config(
        training_iterations = training_iterations,
        storage_path = self.logdir,
        callbacks = callbacks
      )
    except Exception as _:
      raise KeyError(
        "Error: The program could not parse the run config, make sure you "
        "have a stopping criteria defined in the exp config file"
      )
    # tune
    tune_results = self.tuning(
      algo_name = algo_name,
      param_space = param_space,
      tune_config = tune_config,
      run_config = run_config
    )
    # write the configuration and result(s) of the best trial
    self.write_best_trial(
      results = tune_results,
      algo = algo
    )
    experiment_directory = tune_results.experiment_path
    self.logger.log(
      f"Tuning experiment finished successfully, tuning output directory: {experiment_directory}"
    )

  def tuning(
      self,
      algo_name: str = None,
      param_space:dict = None,
      tune_config: tune.TuneConfig = None,
      run_config: air.RunConfig = None
    ):
    # logging & updating progress
    start = datetime.now()
    self.logger.log(f"Tuning --> START")
    self.update_progress_file("experiment_start_timestamp", start.timestamp())
    # runs tuning
    tuner = tune.Tuner(
      algo_name,
      run_config = run_config,
      param_space = param_space,
      tune_config = tune_config,
    )
    results = tuner.fit()
    # logging & updating progress
    end = datetime.now()
    self.update_progress_file("experiment_end_timestamp", end.timestamp())
    experiment_duration = end - start
    self.update_progress_file(
      "experiment_duration_s", experiment_duration.total_seconds()
    )
    self.logger.log(f"experiment took: {experiment_duration}")
    return results

  def write_best_trial(
      self,
      results: tune.ResultGrid = None,
      algo = None
    ):
    # get best hyperparameters
    best_results = results.get_best_result()
    # convert the config to the desired format
    best_results_config = algo.algo_config_generator.to_json(
      best_results.config
    )
    # directory
    best_trial_dir = os.path.join(self.logdir, "complete_config")
    # save the best tune trial config into a json config file
    write_config_file(
      jconfig = best_results_config,
      dirname = best_trial_dir,
      filename = "best_tune_trial_config.json"
    )
    # save the path to the last checkpoint of the best result in the 
    # progress file
    self.update_progress_file(
      "last_checkpoint_dir", best_results.checkpoint.path
    )
    # save the path to the best result directory in the progress file
    self.update_progress_file(
      "best_tune_trial_dir", best_results.path
    )
    # save evaluation results related to the best checkpoint(s)
    for checkpoint_path, result in best_results.best_checkpoints:
      evaluation_metrics = result["evaluation"]
      evaluation_metrics["corresponding_checkpoint"] = checkpoint_path
      self.update_evaluation_metrics_file(
        result["training_iteration"], evaluation_metrics
      )

  def define_stopping_criteria(self):
    """
    Define a `stop()` function to check whether the training loop should be
    terminated, according to the stopping criteria specified in the experiment
    configuration file
    """
    # check that stopping criteria are provided
    if not_defined("stopping_criteria", self.exp_config):
      raise KeyError(
        "`stopping_criteria` must be provided in `exp_config.json`"
      )
    # list possible stopping criteria
    stop_on_max_iter = None
    for key, value in self.exp_config["stopping_criteria"].items():
      if key == "max_iterations":
        stop_on_max_iter = lambda it : it > value
      else:
        raise NotImplementedError(
          f"Stopping criterion `{key}` is not supported"
        )
    self.stop = stop_on_max_iter

  def write_config_files(self):
    """
    Write the environment and experiment configuration files into the
    experiment logdir
    """
    # write environment and experiment configuration files
    super().write_config_files()
    # write tune configuration file
    write_config_file(
      json.dumps(self.tune_config, indent=2),
      os.path.join(self.logdir, "complete_config"),
      "tune_config.json"
    )
