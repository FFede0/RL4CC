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

from ray.rllib.algorithms import AlgorithmConfig
from ray import tune


class PPOConfigGenerator(AlgoConfigGenerator):
  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ):
    super().__init__(logger)
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
    if "rollout_fragment_length" in all_params:
      rfl = all_params["rollout_fragment_length"]
      nw = all_params.get(
        "num_rollout_workers",
        max(self.base_algo_config["num_rollout_workers"], 1)
      )
      if self.is_tuned(rfl):
        all_params["_RL4CC_INTERNALS_num_workers"] = nw
        all_params["train_batch_size"] = tune.sample_from(
          lambda spec: (
            spec.config.rollout_fragment_length * \
              spec.config._RL4CC_INTERNALS_num_workers
          )
        )
      else:
        all_params["train_batch_size"] = rfl * nw
    # sgd batch size
    if "batch_size" in all_params:
      batch_size = all_params.pop("batch_size")
      all_params["sgd_minibatch_size"] = batch_size
    # number of sgd iterations
    if "num_train_batches" in all_params:
      num_batches = all_params.pop("num_train_batches")
      all_params["num_sgd_iter"] = num_batches
  
  def count_sampled_steps(
      self, algo_config: AlgorithmConfig
    ) -> AlgoConfigGenerator.ParameterDomain:
    """
    Counts the number of sampled steps according to the given `AlgorithmConfig`
    """
    # number of rollout workers
    nw = algo_config["num_rollout_workers"]
    # number of collected steps
    for wid in range(1, max(nw, 1) + 1):
      # number of collected steps (per worker)
      rfl = algo_config.get_rollout_fragment_length()
      self.logger.log(
        f"worker {wid}/{max(nw, 1)} collects "
        f"{self.replace_tune_objects(rfl)} step(s)", 
        1
      )
    ncs = self.scale_parameter(rfl, scale_factor = max(nw, 1))
    tbs = algo_config["train_batch_size"]
    self.logger.log(
      f"{self.replace_tune_objects(ncs.value)} step(s) collected to reach "
      f"the required number {self.replace_tune_objects(tbs)}", 
      1
    )
    return ncs
  
  def count_trained_steps(
      self, algo_config: AlgorithmConfig
    ) -> AlgoConfigGenerator.ParameterDomain:
    """
    Counts the number of trained steps according to the given `AlgorithmConfig`
    """
    # number of steps sampled from experience
    sgdbs = algo_config["sgd_minibatch_size"]
    sgdit = algo_config["num_sgd_iter"]
    self.logger.log(
      f"{self.replace_tune_objects(sgdit)} batch(es) of size "
      f"{self.replace_tune_objects(sgdbs)} sampled from experience to train", 
      1
    )
    # check if parameters are coherent
    if self.incompatible_batch_and_sampled_steps(algo_config):
      tbs = algo_config["train_batch_size"]
      raise ValueError(
        f"The training `batch_size` ({self.replace_tune_objects(sgdbs)}) "
        f"must be <= the number of collected steps ({tbs})"
      )
    return self.scale_parameter(sgdbs, scale_factor = sgdit)

  def incompatible_batch_and_sampled_steps(
      self, algo_config: AlgorithmConfig
    ) -> bool:
    if not self.use_tune:
      batch_size = algo_config["sgd_minibatch_size"]
      sampled_steps = algo_config["train_batch_size"]
      return batch_size > sampled_steps
    else:
      batch_size = algo_config["sgd_minibatch_size"]
      sampled_steps = algo_config["train_batch_size"]
      if self.is_tuned(batch_size):
        try:
          batch_size = batch_size.lower
        except Exception:
          batch_size = min(batch_size.categories)
      if self.is_tuned(sampled_steps):
        sampled_steps = self.scale_parameter(
          algo_config["rollout_fragment_length"],
          scale_factor = algo_config["_RL4CC_INTERNALS_num_workers"]
        )
        try:
          sampled_steps = sampled_steps.upper
        except Exception:
          sampled_steps = max(sampled_steps.categories)
      return batch_size > sampled_steps
