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

from ray.rllib.algorithms.dqn.dqn import calculate_rr_weights
from ray.rllib.algorithms import AlgorithmConfig


class DQNConfigGenerator(AlgoConfigGenerator):
  def __init__(self):
    super().__init__()
    self.algo = "DQN"
    # generate default `AlgorithmConfig`
    self.generate_default_config()
    # save a dictionary of algo methods and corresponding parameters
    self.save_algo_methods_dict()
    # algorithm-specific protected/suggested keys
    self._protected_keys += [
      ("training_intensity", "training")
    ]
  
  def convert_training_parameters(self, all_params: dict):
    """
    Defines the appropriate parameters related to the definition and 
    behavior of the policy training algorithm, according to the provided keys
    """
    # train batch size
    batch_size = self.base_algo_config["train_batch_size"]
    if "batch_size" in all_params:
      batch_size = all_params.pop("batch_size")
      all_params["train_batch_size"] = batch_size
    # training intensity
    if "num_train_batches" in all_params:
      num_batches = all_params.pop("num_train_batches")
      # number of sampled steps
      nw = all_params.get(
        "num_rollout_workers",
        max(1, self.base_algo_config["num_rollout_workers"])
      )
      rfl = all_params.get(
        "rollout_fragment_length",
        self.base_algo_config.get_rollout_fragment_length()
      )
      n_sampled_steps = nw * rfl
      # number of trained steps & intensity
      if num_batches > 1:
        n_trained_steps = batch_size * num_batches
        all_params["training_intensity"] = n_trained_steps // n_sampled_steps
      else:
        all_params["training_intensity"] = None
  
  def count_sampled_steps(self, algo_config: AlgorithmConfig) -> int:
    """
    Counts the number of sampled steps according to the given `AlgorithmConfig`
    """
    # number of rollout workers
    nw = algo_config["num_rollout_workers"]
    # proportion between collection and training
    citer, _ = calculate_rr_weights(algo_config)
    # number of collected steps
    ncs = 0
    for wid in range(max(nw, 1)):
      # number of collected steps (per worker)
      rfl = algo_config.get_rollout_fragment_length()
      self.logger.log(f"worker {wid}/{max(nw, 1)} collects {rfl} step(s)")
      ncs += rfl
    self.logger.log(
      f"{ncs} step(s) collected in each of the {citer} collection iteration(s)"
    )
    # wait before start training?
    wait_n_steps = algo_config["num_steps_sampled_before_learning_starts"]
    if wait_n_steps > 0:
      self.logger.log(
        f"{wait_n_steps} steps have to be sampled before learning starts"
      )
    return ncs * citer
  
  def count_trained_steps(self, algo_config: AlgorithmConfig) -> int:
    """
    Counts the number of trained steps according to the given `AlgorithmConfig`
    """
    # proportion between collection and training
    _, titer = calculate_rr_weights(algo_config)
    # number of steps sampled from a replay buffer
    tbs = algo_config["train_batch_size"]
    C = algo_config["replay_buffer_config"]["capacity"]
    self.logger.log(
      f"{titer} batch(es) of size {tbs} sampled from RB (capacity: {C})"
    )
    return tbs * titer
