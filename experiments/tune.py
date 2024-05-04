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
from experiments.base_experiment import BaseExperiment
from algorithms.algorithm import Algorithm
from algorithms.generators.tune_config_generator import TuneConfigGenerator
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from utilities.common import not_defined, write_config_file, load_config_file
from ray import tune, air


from datetime import datetime
import numpy as np
import json
import os

class TuningExperiment(BaseExperiment):
  def __init__(self, exp_config_file: str):
    super().__init__(exp_config_file)

  def run(self,
          callbacks: DefaultCallbacks = None
          ):

    if self.tune_config is None:
      raise FileNotFoundError(
          "In order to run a tune experiment, a tune_config.json file must be provided and indicated in the exp_config.json file"
      )
    else:
      self.tune_config = load_config_file(self.tune_config)
      if self.tune_config is None:
        raise FileNotFoundError(
          "A tune_config.json file is indicated in the exp_config.json file but it could not be found, make sure it exists"
        )
      use_tune = True


    # Get tune params
    tune_config_generator = TuneConfigGenerator()
    tune_config = tune_config_generator.get_tune_config(tune_config=self.tune_config)

    # define algorithm
    algo = Algorithm(
      algo_name=self.exp_config["algorithm"],
      checkpoint_path=self.checkpoint_path,
      env_config=self.env_config,
      ray_config=self.ray_config,
      base_logdir=self.logdir,
      eval_interval=self.evaluation_interval,
      use_tune=use_tune,
    )

    # Write algorithm config file to output dir
    self.logdir =  algo.logdir
    # save experiment configuration files
    self.write_config_files()
    algo.print_algo_config()

    # Prepare Run & Search Space config params
    algo_name = self.exp_config.get("algorithm")
    training_iterations = self.exp_config.get("stopping_criteria", {}).get("max_iterations", 10)
    param_space = algo.algo_config


    try:
      run_config = tune_config_generator.get_run_config(training_iterations=training_iterations,
                                                        storage_path=self.logdir
                                                        )
    except Exception as e:
      raise KeyError("Error: The program could not parse the run config, make sure you have a stopping criteria defined in the exp config file")

    tune_results = self.tuning(algo_name=algo_name,
                               param_space=param_space,
                               tune_config=tune_config,
                               run_config=run_config
                               )

    self.write_best_trial_config(results=tune_results)

    results_df = tune_results.get_dataframe()
    experiment_directory = tune_results.experiment_path
    self.logger.log(f"Tuning experiment finished successfully, tuning output directory: {experiment_directory}")


  def tuning(self,
             algo_name: str = None,
             param_space:dict = None,
             tune_config: tune.TuneConfig = None,
             run_config: air.RunConfig = None
             ):

    # Logging & updating progress
    start = datetime.now()
    self.logger.log(f"Tuning --> START")
    self.update_progress_file("Tuning_start_timestamp", start.timestamp())

    # runs tuning
    tuner = tune.Tuner(algo_name,
                       run_config=run_config,
                       param_space=param_space,
                       tune_config=tune_config,
                       )

    results = tuner.fit()

    # Logging & updating progress
    end = datetime.now()
    self.update_progress_file("experiment_end_timestamp", end.timestamp())
    experiment_duration = end - start
    self.update_progress_file(
      "Tuning_duration_s", experiment_duration.total_seconds()
    )
    self.logger.log(f"Tuning took: {experiment_duration}")

    return results


  def write_best_trial_config(self,
                              results: tune.ResultGrid = None):
    # Get best hyperparameters
    best_results = results.get_best_result()
    best_hyperparameters = best_results.config

    # Save as Json
    best_trial_dir = os.path.join(self.logdir, "complete_config/best_tune_trial_config.json")

    with open(best_trial_dir, 'w') as f:
      json.dump(best_hyperparameters, f, indent=4)

  def validate_experiment_configuration(self):
    super().validate_experiment_configuration()
    # the algorithm name must be provided
    if not_defined("algorithm", self.exp_config):
      raise KeyError(
        "ERROR: `algorithm` is required"
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

  def update_progress_file(self, key: str, value):
    """
    Update the information written in the experiment progress file
    """
    exp_progress = {}
    exp_progress_file = os.path.join(self.logdir, "exp_progress.json")
    # load existing content (if any)
    if os.path.exists(exp_progress_file):
      with open(exp_progress_file, "r") as istream:
        exp_progress = json.load(istream)
    # update
    exp_progress[key] = value
    # write updated file
    with open(exp_progress_file, "w") as ostream:
      ostream.write(json.dumps(exp_progress, indent = 2))

  def update_evaluation_metrics_file(
      self, last_iter: int, evaluation_metrics: dict
    ):
    """
    Save the result of the last evaluation
    """
    # create the serialized dictionary of the last evaluation results
    evaluation = {
      "after_training_iteration": last_iter,
      **self.serialize_evaluation_metrics(evaluation_metrics)
    }
    # write
    evaluation_file = os.path.join(self.logdir, "evaluation.txt")
    with open(evaluation_file, "a") as ostream:
      ostream.write(f"{evaluation}\n")

