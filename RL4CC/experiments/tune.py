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
from RL4CC.utilities.common import load_config_file, write_config_file
from RL4CC.experiments.base_experiment import BaseExperiment
from RL4CC.utilities.common import not_defined, defined
from RL4CC.utilities.logger import Logger
from RL4CC.algorithms.tuner import Tuner

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.result_grid import ResultGrid
from datetime import datetime
import json
import os


class TuningExperiment(BaseExperiment):
  def __init__(self,
               exp_config_file: str = None,
               exp_config: dict = None,
               logger: Logger = Logger(name = "RL4CC")):
    super().__init__(exp_config_file, exp_config, logger)
  
  def validate_experiment_configuration(self):
    super().validate_experiment_configuration()
    # if no checkpoint is provided, the tune configuration is mandatory
    if (
      not_defined("from_checkpoint", self.exp_config) and 
        not_defined("tune_config_file", self.exp_config) and 
          not_defined("tune_config", self.exp_config)
    ):
      raise KeyError(
        "ERROR: provide 'tune_config_file' or 'tune_config' if no previous "
        "checkpoint is given"
      )
    # the tune configuration cannot be simultaneously provided as a file and 
    # as a dictionary
    if (
      defined("tune_config_file", self.exp_config) and 
        defined("tune_config", self.exp_config)
    ):
      raise KeyError(
        "ERROR: 'tune_config_file' or 'tune_config' cannot be both set!"
      )
    # load the tune_config. The user can specify the tune_config via the
    # tune_config_file parameter or directly via the tune_config parameter
    self.tune_config = None
    if defined("tune_config_file", self.exp_config):
      self.tune_config = load_config_file(self.exp_config["tune_config_file"])
    elif defined("tune_config", self.exp_config):
      self.tune_config = self.exp_config["tune_config"]

  def define_checkpoint_config(self):
    """
    Initialize the dictionary storing all configuration parameters related 
    to checkpointing
    """
    super().define_checkpoint_config()
    self.checkpoint_config["checkpoint_at_end"] = True

  def run(self, callbacks: DefaultCallbacks = None):
    # define tuner
    tuner = Tuner(
      checkpoint_path = self.checkpoint_path,
      algo = self.exp_config["algorithm"],
      tune_config = self.tune_config,
      ray_config = self.ray_config,
      env_config = self.env_config,
      checkpoint_config = self.checkpoint_config,
      stopping_criterion = self.stop(),
      eval_interval = self.evaluation_interval,
      storage_path = self.logdir,
      callbacks = callbacks,
      logger = self.logger
    )
    self.logdir = tuner.storage_path
    # save experiment configuration files
    self.write_config_files()
    # tune
    tune_results = self.tuning(tuner)
    # write the configuration and result(s) of the best trial
    self.write_best_trial(tune_results, tuner)
    experiment_directory = tune_results.experiment_path
    self.logger.log(
      "Tuning experiment finished successfully, tuning output "
      f"directory: {experiment_directory}"
    )

  def tuning(self, tuner: Tuner) -> ResultGrid:
    # logging & updating progress
    start = datetime.now()
    self.logger.log(f"Tuning --> START")
    self.update_progress_file("experiment_start_timestamp", start.timestamp())
    # fit
    results = tuner.fit()
    # logging & updating progress
    end = datetime.now()
    self.update_progress_file("experiment_end_timestamp", end.timestamp())
    experiment_duration = end - start
    self.update_progress_file(
      "experiment_duration_s", experiment_duration.total_seconds()
    )
    self.logger.log(f"experiment took: {experiment_duration}")
    # save the tuning results as a dataframe
    results_df = results.get_dataframe()
    results_df.to_csv(
      os.path.join(results.experiment_path, "tuning_results_df.csv"),
      index = False
    )
    return results

  def write_best_trial(self, results: ResultGrid, tuner: Tuner):
    # get best hyperparameters
    best_results = results.get_best_result()
    # convert the config to the desired format
    best_results_config = tuner.algo_config_generator.to_json(
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
    # save evaluation results related to the best checkpoint(s) (if any)
    for checkpoint_path, result in best_results.best_checkpoints:
      if "evaluation" in result:
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
    # list possible stopping criteria
    stop_on_max_iter = lambda : None
    if "stopping_criteria" in self.exp_config:
      for key, value in self.exp_config["stopping_criteria"].items():
        if key == "max_iterations":
          stop_on_max_iter = lambda : {"training_iteration": value}
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