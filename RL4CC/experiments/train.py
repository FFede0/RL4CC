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

from datetime import datetime
import os
import json

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

  def run(self):
    # define algorithm
    self.env_config["logdir"] = self.logdir
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

    if "load_policy_weights" in self.exp_config and self.exp_config["load_policy_weights"]:
      self.logger.log("Loading policy weights", 1)
      algo = self.load_policy_weights(algo)

    self.write_config_files()
    algo.print_algo_config()
    self.logdir = algo.logdir
  
    self.execute_before_training(algo)
    self.training_loop(algo)
    self.execute_after_training(algo)
    
    return algo

  def execute_before_training(self, algo: Algorithm):
    pass

  def on_iteration_start(self, algo: Algorithm, it: int):
    pass

  def on_iteration_end(self, algo: Algorithm, it: int):
    with open(f"{algo.logdir}/exp_progress.json", "r") as f:
      exp_progress = json.load(f)
      if not "custom_metrics" in exp_progress.keys():
        s4air_differences = []
        valid_violations = False
      else:
        if not "average_vm_difference" in exp_progress["custom_metrics"].keys():
          s4air_differences = []
        else:
          s4air_differences = exp_progress["custom_metrics"]["average_vm_difference"]
          
        if not "valid_violations" in exp_progress["custom_metrics"].keys():
          valid_violations = False
        else:
          valid_violations = exp_progress["custom_metrics"]["valid_violations"]
          
    return s4air_differences, valid_violations

  def training_loop(self, algo: Algorithm):
    """
    `Algorithm` training loop
    """
    start = datetime.now()
    self.logger.log(f"training loop --> START", 1)
    self.update_progress_file("experiment_start_timestamp", start.timestamp())
    it = 1
    episode_reward_mean = 0
    s4air_differences = []
    valid_violations = False
    epsilon_reset = self.exp_config.get("epsilon_reset", 0)
    while not self.stop(it, episode_reward_mean, s4air_differences, valid_violations): #TODO: move to custom stopping criteria
      self.on_iteration_start(algo, it)
      # train
      true_it = algo.last_iteration() + 1
      self.logger.log(f"starting iteration {it} ({true_it})", 3)
      result = algo.train()
      episode_reward_mean = result["episode_reward_mean"]
      self.logger.log("iteration completed", 3)
      self.update_progress_file("last_iteration", algo.last_iteration())
      # save checkpoint at the beginning and every `checkpoint_frequency`
      # iterations
      if it == 1 or it % self.checkpoint_config["checkpoint_frequency"] == 0:
        last_chpt_dir = algo.save_checkpoint()
        self.update_progress_file("last_checkpoint_dir", last_chpt_dir)
      # plot results at the beginning and every `plot_interval` iterations
      if it == 1 or it % self.plot_interval == 0:
        self.plot_results(result)
      # save evaluation results every `evaluation_interval` iterations
      if it % self.evaluation_interval == 0:
        print("Evaluating at iteration", it, flush=True)
        self.update_evaluation_metrics_file(
          result["training_iteration"],
          result["evaluation"]
        )
      s4air_differences, valid_violations = self.on_iteration_end(algo, it)
      
      if epsilon_reset != 0 and it % epsilon_reset == 0:
        policy = algo.get_policy()

        policy.exploration.epsilon_schedule.schedule_timesteps = self.ray_config["exploration"]["exploration_config"]["epsilon_schedule"]["schedule_timesteps"]
        policy.exploration.epsilon_schedule.initial_p = self.ray_config["exploration"]["exploration_config"]["epsilon_schedule"]["initial_p"]

        state = policy.get_state()

        exploration_state = state.get("_exploration_state", {})
        exploration_state["cur_epsilon"] = self.ray_config["exploration"]["exploration_config"]["epsilon_schedule"]["initial_p"]
        exploration_state["last_timestep"] = 0

        state["_exploration_state"] = exploration_state

        policy.set_state(state)

      it += 1
    # save last checkpoint
    last_chpt_dir = algo.save_checkpoint()
    self.update_progress_file("last_checkpoint_dir", last_chpt_dir)
    # perform final evaluation
    print("Performing final evaluation", flush=True)
    self.logger.log(f"starting final evaluation", 2)
    self.update_evaluation_metrics_file(
      result["training_iteration"], algo.evaluate()
    )
    self.logger.log(f"final evaluation performed", 2)
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

  def load_policy_weights(self, algo: Algorithm):
    """
    Load policy from given checkpoint
    """

    self.logger.log("Loading policy from checkpoint: {}".format(self.exp_config["load_policy_weights"]), 1)
                    
    old_algo = Algorithm(
      algo_name = self.exp_config["algorithm"],
      checkpoint_path = self.exp_config["load_policy_weights"],
      env_config = self.env_config,
      ray_config = self.ray_config,
      logdir = self.logdir,
      eval_interval = self.evaluation_interval,
      logger = self.logger
    )
    old_algo.build()

    policy_weights = old_algo.get_policy().get_weights()
    algo.get_policy().set_weights(policy_weights)

    return algo