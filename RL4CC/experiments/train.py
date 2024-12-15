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
from RL4CC.utilities.common import not_defined
from RL4CC.utilities.logger import Logger

from ray.rllib.policy.policy import Policy
from datetime import datetime


class TrainingExperiment(BaseExperiment):
  def __init__(
      self,
      exp_config_file: str = None,
      exp_config: dict = None,
      logger: Logger = Logger(name = "RL4CC")
    ):
    super().__init__(exp_config_file, exp_config, logger)
  
  def validate_experiment_configuration(self):
    super().validate_experiment_configuration()
    # check that stopping criteria are provided
    if not_defined("stopping_criteria", self.exp_config):
      raise KeyError(
        "`stopping_criteria` must be provided in `exp_config.json`"
      )
  
  def run(self) -> Policy:
    # define algorithm
    algo = Algorithm(
      algo_name = self.exp_config["algorithm"], 
      checkpoint_path = self.checkpoint_path,
      env_config = self.env_config,
      ray_config = self.ray_config,
      logdir = self.logdir,
      eval_interval = self.evaluation_interval,
      logger = self.logger
    )
    # build (if the algorithm is not loaded from an existing checkpoint)
    if self.checkpoint_path is None:
      algo.build()
    # save logdir
    self.logdir = algo.logdir
    # save experiment configuration files
    self.write_config_files()
    algo.print_algo_config()
    # training loop
    self.execute_before_training(algo)
    self.training_loop(algo)
    self.execute_after_training(algo)
    return algo

  def execute_before_training(self, algo: Algorithm):
    pass

  def on_iteration_start(self, algo: Algorithm, it: int):
    pass

  def on_iteration_end(self, algo: Algorithm, it: int):
    pass

  def training_loop(self, algo: Algorithm):
    """
    `Algorithm` training loop
    """
    start = datetime.now()
    self.logger.log(f"training loop --> START", 1)
    self.update_progress_file("experiment_start_timestamp", start.timestamp())
    it = 1
    while not self.stop(it):
      self.on_iteration_start(algo, it)
      # train
      true_it = algo.last_iteration() + 1
      self.logger.log(f"starting iteration {it} ({true_it})", 3)
      result = algo.train()
      self.logger.log("iteration completed", 3)
      self.update_progress_file("last_iteration", algo.last_iteration())
      # save checkpoint at the beginning and every `checkpoint_frequency` 
      # iterations
      if it == 1 or it % self.checkpoint_config["checkpoint_frequency"] == 0:
        last_chpt_dir = algo.save_checkpoint()
        self.update_progress_file("last_checkpoint_dir", last_chpt_dir)
      # save evaluation results every `evaluation_interval` iterations
      if it % self.evaluation_interval == 0:
        self.update_evaluation_metrics_file(
          result["training_iteration"], 
          result["evaluation"]
        )
      # plot results at the beginning and every `plot_interval` iterations
      if it == 1 or it % self.plot_interval == 0:
        self.plot_results(result)
      self.on_iteration_end(algo, it)
      # move to the next iteration
      it += 1
    # save last checkpoint
    last_chpt_dir = algo.save_checkpoint()
    self.update_progress_file("last_checkpoint_dir", last_chpt_dir)
    # perform final evaluation (if it has not just be performed)
    if (it - 1) % self.evaluation_interval != 0:
      self.logger.log(f"starting final evaluation", 2)
      self.update_evaluation_metrics_file(
        result["training_iteration"], algo.evaluate()
      )
      self.logger.log(f"final evaluation performed", 2)
    else:
      self.logger.log(f"final evaluation already performed during training", 2)
    # stop
    algo.stop()
    end = datetime.now()
    self.update_progress_file("experiment_end_timestamp", end.timestamp())
    self.logger.log("training loop ---> END", 1)
    # last progress update
    experiment_duration = end - start
    avg_time_per_iter = (end - start) / (it - 1)
    self.update_progress_file(
      "experiment_duration_s", experiment_duration.total_seconds()
    )
    self.update_progress_file(
      "avg_time_per_iter_s", avg_time_per_iter.total_seconds()
    )
    self.logger.log(f"training loop took: {experiment_duration}", 1)
    self.logger.log(f"average time per iteration: {avg_time_per_iter}", 1)

  def execute_after_training(self, algo: Algorithm):
    pass
  
  def define_stopping_criteria(self):
    """
    Define a `stop()` function to check whether the training loop should be 
    terminated, according to the stopping criteria specified in the experiment 
    configuration file
    """
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
