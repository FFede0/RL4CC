"""
Copyright 2024 Mohanad Diab, Federica Filippini

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
from RL4CC.algorithms.generators.tune_config_generator import TuneConfigGenerator
from RL4CC.algorithms.generators_factory import ACGfactory
from RL4CC.utilities.logger import Logger

from ray.tune.result_grid import ResultGrid
from ray.tune import Tuner as RayTuner


class Tuner:
  def __init__(
      self,
      algo: str,
      checkpoint_path: str = None,
      tune_config: dict = None,
      ray_config: dict = None,
      env_config: dict = None,
      checkpoint_config: dict = None,
      stopping_criterion: dict = None,
      eval_interval: int = None,
      storage_path: str = None,
      callbacks: list = None,
      logger: Logger = Logger(name="RL4CC-Tuner")
    ):
    self.logger = logger
    self.tune_config_generator = TuneConfigGenerator(logger = self.logger)
    self.algo_config_generator = ACGfactory.create(
      algo, logger = self.logger
    )
    # load the Ray `Tuner` from a checkpoint (if provided)
    if checkpoint_path is not None:
      self.load_checkpoint(checkpoint_path, algo, tune_config)
    # otherwise...
    else:
      if tune_config is None or stopping_criterion is None:
        raise RuntimeError(
          "ERROR: missing tune configuration, algorithm name or stopping "
          "criterion (all mandatory parameters to create a new `Tuner` object)"
        )
      # ...generate AlgorithmConfig
      self.algo_config = self.algo_config_generator.generate_algo_config(
        ray_config = ray_config,
        env_config = env_config,
        eval_interval = eval_interval,
        exp_logdir = storage_path,
        use_tune = True
      )
      # ...generate TunerConfig and RunConfig
      self.tuner_config, self.run_config = self.tune_config_generator.generate(
        tune_config = tune_config,
        stopping_criterion = stopping_criterion,
        storage_path = storage_path,
        callbacks = callbacks,
        checkpoint_config = checkpoint_config
      )
      # ...generate Tuner
      self.tuner = RayTuner(
        algo,
        run_config = self.run_config,
        param_space = self.algo_config.to_dict(),
        tune_config = self.tuner_config,
      )
      self.storage_path = self.run_config.storage_path
      if self.storage_path is not None:
        self.logger.warn(
          "If ray.__version__ < 2.10, temporary files may be stored in "
          f"~/ray_results before being moved to {self.storage_path}. Set the "
          "environment variable `RAY_AIR_LOCAL_CACHE_DIR` to override this"
        )
      else:
        d = self.tuner._local_tuner.get_experiment_checkpoint_dir()
        self.storage_path = d
      self.logger.warn(
        f"`Tuner` created; output directory: {self.storage_path}"
      )
  
  def load_checkpoint(
      self, checkpoint_path: str, algo: str, tune_config: dict = None
    ):
    if RayTuner.can_restore(checkpoint_path):
      restore_config = tune_config if tune_config is not None else {}
      self.tuner = RayTuner.restore(
        # path to previous experiments
        path = checkpoint_path,
        # algorithm
        trainable = algo,
        # resume instructions
        resume_errored = restore_config.get("resume_errored", False),
        restart_errored = restore_config.get("restart_errored", False),
        resume_unfinished = restore_config.get("resume_unfinished", True)
      )
      self.tuner_config = self.tuner._local_tuner._tune_config
      self.run_config = self.tuner._local_tuner.get_run_config()
      self.storage_path = self.run_config.storage_path
    else:
      raise RuntimeError(
        f"Previous experiment {checkpoint_path} cannot be restored"
      )
  
  def fit(self) -> ResultGrid:
    return self.tuner.fit()
