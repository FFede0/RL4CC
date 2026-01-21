"""
Copyright 2026 Federica Filippini

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
from RL4CC.utilities.common import NumpyEncoder,json_to_array_dict,not_defined
from RL4CC.experiments.train import TrainingExperiment
from RL4CC.log_and_report.rl4cc_logger import Logger
from RL4CC.algorithms.algorithm import Algorithm

from datetime import datetime
import json
import os


class FederatedAgentTrain(TrainingExperiment):
  def __init__(self, config, agent_id = "FederatedAgent"):
    super().__init__(config)
    self.iterations_result = []
    self.merged_results = {}
    # in case of first round some operation are not done
    self.first_round = True
    # to keep track of rounds duration
    self.rounds_duration = []
    # set an identifier for the agent
    self.agent_id = agent_id
    # if logging on files, define the local agent logger to log local train 
    # information separately 
    verbose_ = self.logger.verbose
    if self.exp_config["logger"].get("file_streams", False):
      self.logfile = open(
        os.path.join(self.logdir, f"{self.agent_id}.log"), "a"
      )
      self.logger = Logger(
        name=self.agent_id, out_stream=self.logfile, verbose=verbose_
      )
    else:
      # otherwise change the name only
      self.logger = Logger(name=self.agent_id, verbose=verbose_)

  def validate_experiment_configuration(self):
    super().validate_experiment_configuration()
    # define the folder used for federation procedure
    if not_defined("federation_folder", self.exp_config):
      raise KeyError(
        "ERROR: you must define the federation folder in exp_config"
      )
    else:
      self.federation_folder = self.exp_config["federation_folder"]
  
  def on_iteration_start(self, algo: Algorithm, it: int):
    # load the global (aggregated) weights from the previous fed round into 
    # the algorithm model/policy
    aggregated_weights = os.path.join(
      self.federation_folder, f"round_{it-1}", self.agent_id
    )
    if not self.first_round:
      algo.get_policy().set_weights(
        json_to_array_dict(
          os.path.join(aggregated_weights, "agg_weights.json")
        )
      )
      self.logger.log(f"Weights loaded from {aggregated_weights}", 2)
    return algo

  def on_iteration_end(self, algo: Algorithm, it: int):
    # save the latest update (local) weights into the folder used by the 
    # global aggregator
    weights_folder = os.path.join(
      self.federation_folder, f"round_{it}", self.agent_id
    )
    os.makedirs(weights_folder, exist_ok=True)
    weights = self.algo.get_policy().get_weights()
    with open(os.path.join(weights_folder, "weights.json"), 'w') as f:
      json.dump(weights, f, indent = 2, cls=NumpyEncoder)
    self.update_progress_file("last_weights_dir", weights_folder)
    self.logger.log(f"Current weights saved into {weights_folder}", 2)
  
  def local_training_loop(self, algo: Algorithm):
    self.logger.log(f"local training loop --> START", 3)
    it = 1
    while not self.stop({"training_iteration": it}):
      # train
      true_it = algo.last_iteration() + 1
      self.logger.log(f"starting iteration {it} ({true_it})", 4)
      result = algo.train()
      self.logger.log("iteration completed", 4)
      self.update_progress_file("last_iteration", algo.last_iteration())
      # save checkpoint at the beginning and every `checkpoint_frequency` 
      # iterations
      if (
          true_it == 1 or 
            true_it % self.checkpoint_config["checkpoint_frequency"] == 0
        ):
        last_chpt_dir = algo.save_checkpoint()
        self.update_progress_file("last_checkpoint_dir", last_chpt_dir)
      # save evaluation results every `evaluation_interval` iterations
      if true_it % self.evaluation_interval == 0:
        self.update_evaluation_metrics_file(
          result["training_iteration"], 
          result["evaluation"]
        )
      # plot results at the beginning and every `plot_interval` iterations
      if true_it == 1 or true_it % self.plot_interval == 0:
        self.plot_results(result)
      # move to the next iteration
      it += 1
    # save last checkpoint
    last_chpt_dir = algo.save_checkpoint()
    self.update_progress_file("last_checkpoint_dir", last_chpt_dir)
    # perform final evaluation (if it has not just be performed)
    if true_it % self.evaluation_interval != 0:
      self.logger.log(f"starting final evaluation", 4)
      self.update_evaluation_metrics_file(
        result["training_iteration"], algo.evaluate()
      )
      self.logger.log(f"final evaluation performed", 4)
    else:
      self.logger.log(f"final evaluation already performed during training", 4)
    self.logger.log("local training loop ---> END", 3)
    return algo    
  
  def training_loop(self, algo: Algorithm):
    """
    Federated training loop
    """
    start = datetime.now()
    self.logger.log(f"federated training loop --> START", 1)
    self.update_progress_file("experiment_start_timestamp", start.timestamp())
    federation_round = 1
    while not self.stop({"federation_round": federation_round}):
      self.logger.log(f"starting federation round {federation_round}", 2)
      # train
      start_round = datetime.now()
      algo = self.on_iteration_start(algo, federation_round)
      algo = self.local_training_loop(algo)
      self.on_iteration_end(algo, federation_round)
      end_round = datetime.now()
      self.logger.log("round completed", 2)
      self.update_progress_file("last_federation_round", federation_round)
      round_duration = (end_round - start_round).total_seconds()
      self.logger.log(f"local training duration: {round_duration:.2f} s", 2)
      # progress update
      self.rounds_duration.append(round_duration)
      self.update_progress_file("rounds_duration", self.rounds_duration)
      self.first_round = False
      # move to the next iteration
      federation_round += 1
    # stop
    algo.stop()
    end = datetime.now()
    self.update_progress_file("experiment_end_timestamp", end.timestamp())
    self.logger.log("training loop ---> END", 1)
    # last progress update
    experiment_duration = end - start
    avg_time_per_iter = (end - start) / (federation_round - 1)
    self.update_progress_file(
      "experiment_duration_s", experiment_duration.total_seconds()
    )
    self.update_progress_file(
      "avg_time_per_iter_s", avg_time_per_iter.total_seconds()
    )
    self.logger.log(f"training loop took: {experiment_duration}", 1)
    self.logger.log(f"average time per iteration: {avg_time_per_iter}", 1)
  
  def define_stopping_criteria(self):
    """
    Define a `stop()` function to check whether the training loop should be 
    terminated, according to the stopping criteria specified in the experiment 
    configuration file
    """
    if not_defined(
        "max_federation_rounds", self.exp_config["stopping_criteria"]
      ) or not_defined(
        "max_iterations", self.exp_config["stopping_criteria"]
      ):
      raise KeyError("Missing stopping criteria")
    self.max_federation_rounds = self.exp_config["stopping_criteria"][
      "max_federation_rounds"
    ]
    self.max_iterations = self.exp_config["stopping_criteria"][
      "max_iterations"
    ]
    self.iterations_before_start = self.exp_config.get(
      "n_train_iterations_before_federation_starts", 0
    )
    def stop_function(params: dict):
      # -- define the true maximum number of iterations
      max_it = self.max_iterations
      if self.first_round:
        max_it += self.iterations_before_start
      # -- check conditions
      if "federation_round" in params:
        if params["federation_round"] > self.max_federation_rounds:
          return True
      if "training_iteration" in params:
        if params["training_iteration"] > max_it:
          return True
      return False
    self.stop = stop_function
