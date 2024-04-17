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
from algorithms.generators.algo_config_generator import AlgoConfigGenerator

from ray.rllib.algorithms import AlgorithmConfig


class PPOConfigGenerator(AlgoConfigGenerator):
  def __init__(self):
    super().__init__()
    self.algo = "PPO"
    # generate default `AlgorithmConfig`
    self.generate_default_config()
    # save a dictionary of algo methods and corresponding parameters
    self.save_algo_methods_dict()
    # algorithm-specific protected/suggested keys
    self._protected_keys += [
      ("sgd_minibatch_size", "training"),
      ("num_sgd_iter", "training")
    ]
  
  def convert_training_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to the definition and 
    behavior of the policy training algorithm, according to the provided keys
    """
    # train batch size
    n_steps = self.base_algo_config["train_batch_size"]
    if "rollout_fragment_length" in all_params:
      nw = all_params.get(
        "num_rollout_workers",
        max(self.base_algo_config["num_rollout_workers"], 1)
      )
      n_steps = nw * all_params["rollout_fragment_length"]
      all_params["train_batch_size"] = n_steps
      self.logger.log(
        f"{nw} rollout workers will collect overall {n_steps} steps"
      )
    # sgd batch size
    if "batch_size" in all_params:
      batch_size = all_params.pop("batch_size")
      all_params["sgd_minibatch_size"] = batch_size
      self.logger.log(
        f"training batches will have size: {batch_size}"
      )
    # number of sgd iterations
    if "num_train_batches" in all_params:
      num_batches = all_params.pop("num_train_batches")
      all_params["num_sgd_iter"] = num_batches
      self.logger.log(
       f"{num_batches} batches will be extracted for training"
      )
  
  def validate_collection_and_training_size(
      self, algo_config: AlgorithmConfig
    ):
    """
    Computes the number of collected and trained steps according to the 
    given `AlgorithmConfig`
    """
    self.logger.breakline()
    self.logger.log(
      f"*** collected/trained steps in each `{self.algo}.training_step()` ***"
    )
    # number of rollout workers
    nw = algo_config["num_rollout_workers"]
    # number of collected steps
    ncs = 0
    for wid in range(max(nw, 1)):
      # number of collected steps (per worker)
      rfl = algo_config.get_rollout_fragment_length()
      self.logger.log(f"worker {wid}/{max(nw, 1)} collects {rfl} step(s)")
      ncs += rfl
    tbs = algo_config["train_batch_size"]
    self.logger.log(
      f"{ncs} step(s) collected to reach the required number {tbs}"
    )
    # number of steps sampled from experience
    sgdbs = algo_config["sgd_minibatch_size"]
    sgdit = algo_config["num_sgd_iter"]
    self.logger.log(
      f"{sgdit} batch(es) of size {sgdbs} sampled from experience to train"
    )
    self.logger.breakline()
    # check if parameters are coherent
    if sgdbs > tbs:
      msg = f"The training `batch_size` ({sgdbs}) must be <= the number of "
      raise ValueError(
        msg + f"collected steps ({tbs})"
      )
    # check if the `training_step` function will be called more than once
    self.check_num_training_step_calls(algo_config)