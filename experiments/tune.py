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
from RL4CC.experiments.base_experiment import BaseExperiment
from RL4CC.algorithms.algorithm import Algorithm
from utilities.common import not_defined
from ray import tune, air
from RL4CC.algorithms.generators.dqn_config_generator import calculate_rr_weights

from datetime import datetime
import numpy as np
import json
import os

class TuningExperiment(BaseExperiment):
  def __init__(self, exp_config_file: str):
    super().__init__(exp_config_file)

  def run(self):
    # define algorithm
    algo = Algorithm(
      algo_name=self.exp_config["algorithm"],
      checkpoint_path=self.checkpoint_path,
      env_config=self.env_config,
      ray_config=self.ray_config,
      base_logdir=self.logdir,
      eval_interval=self.evaluation_interval
    )
    # TODO (Mohanad): Uncomment the following calls once the error is solved
    #self.logdir = algo.logdir
    # save experiment configuration files
    #self.write_config_files()
    #algo.print_algo_config()

    tune_results = self.tuning(algo)


  def tuning(self, algo:Algorithm):
    # runs tuning
    # TODO (Mohanad): The following params need to be fetched from the tune config generator
    algo_name = self.exp_config.get("algorithm")
    training_iterations = self.exp_config.get("stopping_criteria", {}).get("max_iterations")
    num_samples = self.ray_config.get("tune_config", {}).get("num_tune_trials")
    param_space = algo.algo_config


    start = datetime.now()
    self.logger.log(f"Tuning --> START")
    self.update_progress_file("Tuning_start_timestamp", start.timestamp())

    tune_config = tune.TuneConfig(num_samples=num_samples,
                                      metric="episode_reward_mean",
                                      mode="max",
                                      # TODO (Mohanad): implement the logic for the definition for the search algo
                                      search_alg=self.search_algorithm,
                                      scheduler=self.scheduler,
                                      )

    run_config = air.RunConfig(name=algo_name,
                               verbose=1,
                               stop={"training_iteration": training_iterations},
                               log_to_file=True
                               )

    results = tune.Tuner(algo_name,
                         run_config=run_config,
                         param_space=param_space,
                         tune_config=tune_config,
                         ).fit()

    return results



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

