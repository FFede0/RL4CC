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
from RL4CC.algorithms.generators.algo_config_generator import AlgoConfigGenerator
from RL4CC.utilities.logger import Logger

from ray.rllib.algorithms.dqn.dqn import calculate_rr_weights
from ray.rllib.algorithms import AlgorithmConfig
from typing import Tuple
from ray import tune


class DQNConfigGenerator(AlgoConfigGenerator):
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ):
    super().__init__(logger)
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
    # compute training intensity
    if "num_train_batches" in all_params:
      # number of train batches
      num_batches = all_params.pop("num_train_batches")
      all_params["_RL4CC_INTERNALS_num_train_batches"] = num_batches
      # number of rollout workers and collected steps per worker
      nw = max(
        1, 
        all_params.get(
          "num_rollout_workers",
          self.base_algo_config["num_rollout_workers"]
        ) + 1
      )
      rfl = all_params.get(
        "rollout_fragment_length",
        self.base_algo_config.get_rollout_fragment_length()
      )
      all_params["_RL4CC_INTERNALS_num_workers"] = nw
      # check whether any parameter is being tuned
      if any(self.is_tuned(k) for k in [rfl, num_batches, batch_size]):
        # if so, the training intensity depends on the sampled values
        all_params["training_intensity"] = tune.sample_from(
          lambda spec: (
            spec.config.train_batch_size * \
              spec.config._RL4CC_INTERNALS_num_train_batches
          ) / (
            spec.config.rollout_fragment_length * \
              spec.config._RL4CC_INTERNALS_num_workers
          )
        )
      elif num_batches > 1:
        # otherwise, if more batches should be collected, the training 
        # intensity is the ratio between the number of trained and collected 
        # steps
        n_sampled_steps = rfl * nw
        n_trained_steps = batch_size * num_batches
        all_params["training_intensity"] = n_trained_steps / n_sampled_steps
      else:
        # if only one batch is considered, let Ray figure it out
        all_params["training_intensity"] = None
    # check potential issues related to the presence of custom models
    if "model" in all_params and "custom_model" in all_params["model"]:
      if "hiddens" in all_params and len(all_params["hiddens"]) > 0:
        self.logger.warn(
          "The initialization may fail if the custom network size and "
          f"`hiddens` ({all_params['hiddens']}) are not compatible"
        )
  
  def count_sampled_steps(
      self, algo_config: AlgorithmConfig
    ) -> AlgoConfigGenerator.ParameterDomain:
    """
    Counts the number of sampled steps according to the given `AlgorithmConfig`
    """
    # number of rollout workers
    nw = algo_config["num_rollout_workers"]
    # proportion between collection and training
    citer, _ = self.calculate_rr_weights(algo_config)
    # number of collected steps
    for wid in range(1, max(nw, 1) + 1):
      # number of collected steps (per worker)
      rfl = algo_config.get_rollout_fragment_length()
      self.logger.log(
        f"worker {wid}/{max(nw, 1)} collects "
        f"{self.replace_tune_objects(rfl)} step(s)", 
        1
      )
    ncs, _, _ = self.scale_parameter(rfl, scale_factor = max(nw, 1))
    self.logger.log(
      f"{self.replace_tune_objects(ncs)} step(s) collected in each "
      f"of the {citer} collection iteration(s)",
      1
    )
    # wait before start training?
    wait_n_steps = algo_config["num_steps_sampled_before_learning_starts"]
    if self.is_tuned(wait_n_steps) or wait_n_steps > 0:
      wait_n_steps = self.replace_tune_objects(wait_n_steps)
      self.logger.log(
        f"{wait_n_steps} steps have to be sampled before learning starts", 1
      )
    return self.scale_parameter(ncs, scale_factor = citer)
  
  def count_trained_steps(self, algo_config: AlgorithmConfig) -> int:
    """
    Counts the number of trained steps according to the given `AlgorithmConfig`
    """
    # proportion between collection and training
    _, titer = self.calculate_rr_weights(algo_config)
    # number of steps sampled from a replay buffer
    tbs = algo_config["train_batch_size"]
    C = self.replace_tune_objects(
      algo_config["replay_buffer_config"]["capacity"]
    )
    self.logger.log(
      f"{self.replace_tune_objects(titer)} batch(es) of size "
      f"{self.replace_tune_objects(tbs)} sampled "
      f"from RB (capacity: {C})", 
      1
    )
    return self.scale_parameter(tbs, scale_factor = titer)
  
  def calculate_rr_weights(
      self, 
      algo_config: AlgorithmConfig
    ) -> Tuple[int, int]:
    citer = 1
    titer = 1
    # if Ray Tune is not used, the numbers can be computed directly
    if not self.use_tune:
      citer, titer = calculate_rr_weights(algo_config)
    # otherwise, we may have to consider domain limits
    else:
      try:
        citer, titer = calculate_rr_weights(algo_config)
      except TypeError:
        if "training_intensity" in algo_config:
          titer = algo_config["_RL4CC_INTERNALS_num_train_batches"]
    return citer, titer
