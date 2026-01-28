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
from RL4CC.experiments.federated_train import FederatedTrainingExperiment
from RL4CC.utilities.common import NumpyEncoder, defined
from RL4CC.log_and_report.rl4cc_logger import Logger
from RL4CC.algorithms.algorithm import Algorithm

from datetime import datetime
import numpy as np
import json
import os


class GossipTrainingExperiment(FederatedTrainingExperiment):
  def __init__(
      self,
      exp_config_file: str = None,
      exp_config: dict = None,
      logger: Logger = Logger(name = "RL4CC")
    ):
    super().__init__(exp_config_file, exp_config, logger)
    self.rng = np.random.default_rng(self.exp_config.get("exp_seed", 4850))
  
  def generate_logdir(self, base_logdir: str, algo: str, env_name: str) -> str:
    """
    Generate the experiment `logdir` if an appropriate parameter is provided
    """
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    self.logdir = os.path.join(
      os.path.abspath(base_logdir), f"{algo}gossip_{env_name}_{now}"
    )
    os.makedirs(self.logdir, exist_ok=True)
  
  def validate_experiment_configuration(self):
    super().validate_experiment_configuration()
    # check that the agents adjacency matrix is provided in the environment 
    # configuration
    if not defined("neighborhood", self.env_config):
      raise KeyError(
        "The agents adjacency matrix must be defined in the environment"
      )
    else:
      self.neighborhood = self.env_config["neighborhood"]
    # check that the number of shared model in each round is not larger than
    # the number of neighbors
    self.n_models_to_share = self.exp_config.get("n_models_to_share", 1)
    if any([
        self.n_models_to_share > len(neigh) \
          for neigh in self.neighborhood.values()
      ]):
      raise ValueError(
        "The number of models to share should be <= the number of neighbors"
      )
  
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
    pairs = {}
    agg_weights = {}
    for agent, neighbors in self.neighborhood.items():
      # -- extract groups to gossip
      pairs[agent] = self.rng.choice(
        neighbors, size = self.n_models_to_share, replace = False
      ).tolist()
      # -- aggregate
      agg_weigths_agent = self.aggregate(
        {n: weights[n] for n in pairs[agent]}, 
        pairs[agent], 
        self.networks_to_aggregate, 
        self.private_layers
      )
      agg_weights[agent] = agg_weigths_agent[pairs[agent][0]]
    # save the aggregated weights
    with open(os.path.join(weights_folder, "agg_weights.json"), "w") as ost:
      json.dump(
        agg_weights, ost, indent = 2, cls = NumpyEncoder, sort_keys = True
      )
    # save the pairs of aggregators
    with open(
        os.path.join(weights_folder, "who_receives_from_whom.json"), "w"
      ) as ost:
      json.dump(pairs, ost, indent = 2, sort_keys = True)
