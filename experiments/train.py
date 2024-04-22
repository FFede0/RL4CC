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
from environment.environments_factory import EnvironmentsFactory
from callbacks.callbacks_factory import CallbacksFactory
from experiments.base_experiment import BaseExperiment
from algorithms.algorithm import Algorithm
from utilities.common import not_defined

from datetime import datetime
import numpy as np
import json
import os

class TrainingExperiment(BaseExperiment):
  def __init__(
      self, 
      exp_config_file: str,
      environments_factory = EnvironmentsFactory,
      callbacks_factory = CallbacksFactory
    ):
    super().__init__(exp_config_file, environments_factory, callbacks_factory)
  
  def validate_experiment_configuration(self):
    super().validate_experiment_configuration()
    # the algorithm name must be provided
    if not_defined("algorithm", self.exp_config):
      raise KeyError(
        "ERROR: `algorithm` is required"
      )
  
  def run(self):
    # define algorithm
    algo = Algorithm(
      algo_name = self.exp_config["algorithm"], 
      checkpoint_path = self.checkpoint_path,
      env_config = self.env_config,
      ray_config = self.ray_config,
      base_logdir = self.logdir,
      eval_interval = self.evaluation_interval,
      environments_factory = self.environments_factory,
      callbacks_factory = self.callbacks_factory
    )
    self.logdir = algo.logdir
    # save experiment configuration files
    self.write_config_files()
    algo.print_algo_config()
    # train
    self.training_loop(algo)
  
  def training_loop(self, algo: Algorithm):
    """
    `Algorithm` training loop
    """
    start = datetime.now()
    self.logger.log(f"training loop --> START")
    self.update_progress_file("experiment_start_timestamp", start.timestamp())
    it = 1
    while not self.stop(it):
      # train
      true_it = algo.last_iteration() + 1
      self.logger.log(f"starting iteration {it} ({true_it})", 2)
      result = algo.train()
      self.logger.log("iteration completed", 2)
      self.update_progress_file("last_iteration", algo.last_iteration())
      # save checkpoint at the beginning and every `checkpoint_interval` 
      # iterations
      if it == 1 or it % self.checkpoint_interval == 0:
        last_chpt_dir = algo.save_checkpoint()
        self.update_progress_file("last_checkpoint_dir", last_chpt_dir)
      # plot results at the beginning and every `plot_interval` iterations
      if it == 1 or it % self.plot_interval == 0:
        self.plot_results(result)
      # save evaluation results every `evaluation_interval` iterations
      if it % self.evaluation_interval == 0:
        self.update_evaluation_metrics_file(
          result["training_iteration"], 
          result.evaluation_metrics
        )
      # move to the next iteration
      it += 1
    # save last checkpoint
    last_chpt_dir = algo.save_checkpoint()
    self.update_progress_file("last_checkpoint_dir", last_chpt_dir)
    # perform final evaluation
    self.logger.log(f"starting final evaluation", 1)
    self.update_evaluation_metrics_file(
      result["training_iteration"], algo.evaluate()
    )
    self.logger.log(f"final evaluation performed", 1)
    # stop
    algo.stop()
    end = datetime.now()
    self.update_progress_file("experiment_end_timestamp", end.timestamp())
    self.logger.log("training loop ---> END")
    # last progress update
    experiment_duration = end - start
    avg_time_per_iter = (end - start) / (it - 1)
    self.update_progress_file(
      "experiment_duration_s", experiment_duration.total_seconds()
    )
    self.update_progress_file(
      "avg_time_per_iter_s", avg_time_per_iter.total_seconds()
    )
    self.logger.log(f"training loop took: {experiment_duration}")
    self.logger.log(f"average time per iteration: {avg_time_per_iter}")
  
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
