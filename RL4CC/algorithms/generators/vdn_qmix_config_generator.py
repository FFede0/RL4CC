"""
Copyright 2026

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
from RL4CC.log_and_report.rl4cc_logger import Logger

from ray.rllib.algorithms import AlgorithmConfig


class _VDNQMIXConfigGeneratorBase(AlgoConfigGenerator):
  """Shared helpers for VDN / QMIX config generators."""

  default_mixer = "vdn"

  def generate_algo_config(
      self,
      env_config: dict,
      ray_config: dict = None,
      exp_logdir: str = None,
      eval_interval: int = None,
      use_tune: bool = False,
      multiagent: bool = False
    ) -> AlgorithmConfig:
    return super().generate_algo_config(
      env_config,
      ray_config=ray_config,
      exp_logdir=exp_logdir,
      eval_interval=eval_interval,
      use_tune=use_tune,
      multiagent=True,
    )

  def process_config_parameters(
      self,
      ray_config: dict,
      env_config: dict,
      exp_logdir: str = None,
      eval_interval: int = None
    ) -> dict:
    all_params = super().process_config_parameters(
      ray_config,
      env_config,
      exp_logdir=exp_logdir,
      eval_interval=eval_interval,
    )
    if "mixer" not in all_params:
      all_params["mixer"] = self.default_mixer
    all_params.setdefault("lr", 5e-4)
    all_params.setdefault("gamma", 0.99)
    all_params.setdefault("batch_size", 64)
    all_params.setdefault("target_network_update_freq", 500)
    all_params.setdefault("grad_clip", 10.0)
    all_params.setdefault("episodes_per_iter", 2)
    all_params.setdefault("gradient_steps_per_iter", 30)
    all_params.setdefault("replay_buffer_capacity", 100000)
    all_params.setdefault("epsilon_start", 1.0)
    all_params.setdefault("epsilon_end", 0.05)
    all_params.setdefault("epsilon_decay_iters", 10000)
    all_params.setdefault("epsilon_decay_power", 1.0)
    if self.default_mixer == "qmix":
      all_params.setdefault("mixing_embed_dim", 32)

    if "model" not in all_params:
      all_params["model"] = {}
    all_params["model"].setdefault("custom_model", "value_decomp_q")
    all_params["model"].setdefault("fcnet_hiddens", [256, 128])
    all_params["model"].setdefault("fcnet_activation", "relu")
    return all_params

  def convert_training_parameters(self, all_params: dict):
    if "batch_size" in all_params:
      all_params["train_batch_size"] = all_params.pop("batch_size")
    if "replay_buffer_config" in all_params:
      rb = all_params.pop("replay_buffer_config") or {}
      if isinstance(rb, dict) and "capacity" in rb:
        all_params.setdefault("replay_buffer_capacity", rb["capacity"])

  def count_sampled_steps(self, algo_config: AlgorithmConfig):
    episodes = algo_config.get("episodes_per_iter", 2)
    self.logger.log(
      f"{self.replace_tune_objects(episodes)} episode(s) collected "
      f"per training iteration (VDN/QMIX local loop)",
      1,
    )
    return self.ParameterDomain(value=episodes, lower=episodes, upper=episodes)

  def count_trained_steps(self, algo_config: AlgorithmConfig):
    tbs = algo_config.get("train_batch_size", 64)
    nsteps = algo_config.get("gradient_steps_per_iter", 30)
    self.logger.log(
      f"{self.replace_tune_objects(nsteps)} gradient step(s) of batch size "
      f"{self.replace_tune_objects(tbs)} per iteration",
      1,
    )
    return self.scale_parameter(tbs, scale_factor=nsteps)


class VDNConfigGenerator(_VDNQMIXConfigGeneratorBase):
  """Config generator for algorithm name ``VDN`` (MAPPO-style init order)."""

  default_mixer = "vdn"

  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ):
    super().__init__(logger)
    self.algo = "VDN"
    self.generate_default_config()
    self.save_algo_methods_dict()
    # Keep mixer knobs as ordinary training keys (not protected), so they can
    # be combined with RL4CC suggested keys like ``batch_size``.


class QMIXConfigGenerator(_VDNQMIXConfigGeneratorBase):
  """Config generator for algorithm name ``QMIX``."""

  default_mixer = "qmix"

  def __init__(
      self, logger: Logger = Logger(name="RL4CC-AlgoConfigGenerator")
    ):
    super().__init__(logger)
    self.algo = "QMIX"
    self.generate_default_config()
    self.save_algo_methods_dict()
