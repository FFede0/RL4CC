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
from RL4CC.utilities.common import NumpyEncoder, json_to_array_dict
from RL4CC.utilities.common import defined, not_defined
from RL4CC.experiments.train import TrainingExperiment
from RL4CC.log_and_report.rl4cc_logger import Logger
from RL4CC.algorithms.algorithm import Algorithm

from datetime import datetime
import numpy as np
import json
import os


class FederatedTrainingExperiment(TrainingExperiment):
  def __init__(
      self,
      exp_config_file: str = None,
      exp_config: dict = None,
      logger: Logger = Logger(name = "RL4CC")
    ):
    super().__init__(exp_config_file, exp_config, logger)
    self.iterations_result = []
    self.merged_results = {}
    # in case of first round some operation are not done
    self.first_round = True
    # to keep track of rounds duration
    self.rounds_duration = []
    # list of networks to aggregate and list of private layers
    self.networks_to_aggregate = self.exp_config.get(
      "networks_to_aggregate", ["all"]
    )
    self.private_layers = self.exp_config.get("private_layers", None)
  
  def generate_logdir(self, base_logdir: str, algo: str, env_name: str) -> str:
    """
    Generate the experiment `logdir` if an appropriate parameter is provided
    """
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    self.logdir = os.path.join(
      os.path.abspath(base_logdir), f"{algo}fed_{env_name}_{now}"
    )
    os.makedirs(self.logdir, exist_ok=True)

  def validate_experiment_configuration(self):
    super().validate_experiment_configuration()
    # check that networks_to_aggregate and private_layers are consistent
    if defined(
        "networks_to_aggregate", self.exp_config
      ) and defined("private_layers", self.exp_config):
      nta = self.exp_config["networks_to_aggregate"]
      pl = self.exp_config["private_layers"]
      if any([l in pl for l in nta]):
        raise ValueError(
          "Overlapping elements in `networks_to_aggregate` "
          "and `private_layers`"
        )
  
  def on_iteration_start(self, algo: Algorithm, it: int):
    # load the global (aggregated) weights from the previous fed round into 
    # the algorithm model/policy
    if not self.first_round:
      aggregated_weights = os.path.join(
        self.federation_folder, f"round_{it-1}"
      )
      algo.set_weights(
        json_to_array_dict(
          os.path.join(aggregated_weights, "agg_weights.json")
        )
      )
      self.logger.log(f"Weights loaded from {aggregated_weights}", 2)
    else:
      # in the first federation round, define folder to save weights
      self.federation_folder = os.path.join(self.logdir, "federation_folder")
    return algo

  def on_iteration_end(self, algo: Algorithm, it: int):
    # save the latest update (local) weights into the folder used by the 
    # global aggregator
    weights_folder = os.path.join(self.federation_folder, f"round_{it}")
    os.makedirs(weights_folder, exist_ok = True)
    weights = algo.get_weights()
    with open(os.path.join(weights_folder, "weights.json"), "w") as ost:
      json.dump(weights, ost, indent = 2, cls = NumpyEncoder, sort_keys = True)
    self.update_progress_file("last_weights_dir", weights_folder)
    self.logger.log(
      f"Current weights saved into {weights_folder}; start aggregation", 2
    )
    # aggregate weights
    agg_weights = self.aggregate(
      weights, weights.keys(), self.networks_to_aggregate, self.private_layers
    )
    # save the aggregated weights
    with open(os.path.join(weights_folder, "agg_weights.json"), "w") as ost:
      json.dump(
        agg_weights, ost, indent = 2, cls = NumpyEncoder, sort_keys = True
      )
  
  def aggregate(
      self, 
      weights: dict, 
      agents: list, 
      networs_to_aggregate: list = ["all"],
      private_layers: list = None
    ):
    # find all keys that appear in any model
    network_layers = set().union(*[weights[ag].keys() for ag in agents])
    local_network_layer = {ag: {} for ag in agents}
    shared_network_layer = []
    for layer_key in network_layers:
      # extract values from agents that have this layer_key
      layer_weights = [weights[ag].get(layer_key, None) for ag in agents]
      # check networks to aggregate
      if networs_to_aggregate != ["all"]:
        if all(
            not layer_key.startswith(prefix) for prefix in networs_to_aggregate
          ):
          for ag, v in zip(agents, layer_weights):
            local_network_layer[ag][layer_key] = v
          continue
      # private layers are local only
      if private_layers is not None:
        if any(layer_key.startswith(prefix) for prefix in private_layers):
          for ag, v in zip(agents, layer_weights):
            local_network_layer[ag][layer_key] = v
          continue
      # check if layer_key is present for all agents
      if any(v is None for v in layer_weights):
        # -- not aggregatable; store individually
        for ag, v in zip(agents, layer_weights):
          if v is not None:
            local_network_layer[ag][layer_key] = v
        continue
      # check if shapes are consistent
      shapes = [v.shape for v in layer_weights]
      if len(set(shapes)) != 1:
        # -- inconsistent shapes; store individually
        for ag, v in zip(agents, layer_weights):
          local_network_layer[ag][layer_key] = v
        continue
      # aggregatable
      shared_network_layer.append(layer_key)
    # aggregate the weights of the shared part of the network
    aggregated_shared = self.aggregate_shared_weights(
      weights, agents, shared_network_layer
    )
    # save aggregated weights per agent
    aggregated_weights = {}
    for ag in agents:
      aggregated_weights[ag] = {}
      # -- add shared averaged parameters
      aggregated_weights[ag].update(aggregated_shared)
      # -- add agent-specific parameters
      aggregated_weights[ag].update(local_network_layer[ag])
    return aggregated_weights
  
  def aggregate_shared_weights(
      self, weights: dict, agents: list, shared_network_layer: list
    ):
    """
    Average shared weights; override this to implement other aggregation rules
    """
    aggregated_shared = {}
    for layer_key in shared_network_layer:
      stacked = np.stack([weights[ag][layer_key] for ag in agents], axis = 0)
      aggregated_shared[layer_key] = np.mean(stacked, axis = 0)
    return aggregated_shared
  
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
        last_chpt_dir = self.save_checkpoint(algo)
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
    last_chpt_dir = self.save_checkpoint(algo)
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
      # train and aggregate
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
