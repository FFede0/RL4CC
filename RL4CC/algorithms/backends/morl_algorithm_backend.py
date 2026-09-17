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
from RL4CC.algorithms.generators.morl_algorithm_generatory import (
  MORLAlgorithmGenerator
)
from RL4CC.algorithms.algorithm_backend import AlgorithmBackend
from RL4CC.log_and_report.rl4cc_logger import Logger

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.weights import equally_spaced_weights
import pandas as pd
import numpy as np
import wandb
import torch
import os


class MORLAlgorithmBackend(AlgorithmBackend):
  def __init__(
      self,
      generator: MORLAlgorithmGenerator,
      checkpoint_path: str = None,
      env_config: dict = None,
      learner_config: dict = None,
      logdir: str = None,
      eval_interval: int = None,
      logger: Logger = Logger(name = "RL4CC-MORLAlgorithmBackend")
    ):
    self.logger = logger
    self.generator = generator
    self.logdir = logdir
    if env_config is None:
      raise RuntimeError(
        "ERROR: no environment configuration provided"
      )
    # ...initialize algorithm
    self.algo_config = self.generator.generate(
      env_config = env_config,
      learner_config = learner_config,
      exp_logdir = logdir,
      eval_interval = eval_interval
    )
    self.algo, self.eval_env, self.additional_params = self.algo_config
    self.evaluation_config = {
      "evaluation_interval": eval_interval,
      **self.additional_params.pop("evaluation")
    }
    # load the algorithm from a checkpoint (if provided)
    # NOTE: the algorithm must be initialized first!
    if checkpoint_path is not None:
      self.load_checkpoint(checkpoint_path)

  def build(self):
    """
    MORL-Baselines algorithms are instantiated by their generators,
    so there is no separate build step; however, build() re-starts the 
    iteration counter
    """
    self.iteration = 1
    return self.algo

  def train(self) -> dict:
    """
    Perform one training iteration
    """
    # train
    with WandbMetricsCollector() as collector:
      self.algo.train(
        eval_env = None,            # no automatic evaluation
        **self.additional_params,
      )
    self.iteration += 1
    return self.update_train_results(collector.records)

  def evaluate(self) -> dict:
    """
    Evaluate the MORL policy over the configured evaluation weights.
    """
    eval_weights = equally_spaced_weights(
      self.eval_env.env.reward_dim, 
      n = self.evaluation_config["num_eval_weights_for_front"]
    )
    num_eval_episodes_for_front = self.evaluation_config[
      "num_eval_episodes_for_front"
    ]
    results = []
    metrics = []
    current_front = []
    for ew in eval_weights:
      res = self.algo.policy_eval(
        self.eval_env, 
        weights = ew, 
        num_episodes = num_eval_episodes_for_front, 
        log = False
      )
      results.append(list(res))
      current_front.append(res[3])
    with WandbMetricsCollector() as collector:
      log_all_multi_policy_metrics(
        current_front = current_front,
        hv_ref_point = self.additional_params["ref_point"],
        reward_dim = self.eval_env.env.reward_dim,
        global_step = self.algo.global_step,
        n_sample_weights = self.evaluation_config[
          "num_eval_weights_for_eval"
        ],
        ref_front = self.evaluation_config.get("known_pareto_front")
      )
    metrics = []
    for el in collector.records:
      if "eval/front" not in el:
        metrics.append(el)
      else:
        d = {}
        for k,v in el.items():
          if k != "eval/front":
            d[k] = v
          else:
            d[k] = v.get_dataframe().to_dict(orient="records")
        metrics.append(d)
    return {
      "eval_weights": eval_weights,
      "eval_res": results,
      "current_front": current_front,
      "eval_metrics": metrics
    }

  def compute_single_action(
      self,
      obs,
      explore: bool = False,
      weight: np.array = None
    ):
    if weight is None:
      raise ValueError("A weight vector is required for MORL algorithms")
    # possibly random action
    if explore:
      return self.algo.act(
        torch.as_tensor(obs),
        torch.as_tensor(weight)
      )
    # greedy action
    return self.algo.max_action(
      torch.as_tensor(obs),
      torch.as_tensor(weight)
    )

  def stop(self):
    """
    Release MORL-Baselines resources.
    """
    # MORL-Baselines does not expose a Ray-like `stop()`.
    # Closing the wandb run is sufficient when logging is enabled.
    if getattr(self.algo, "log", False):
      try:
        self.algo.close_wandb()
      except AttributeError:
        pass

  def last_iteration(self) -> int:
    return self.iteration - 1

  def save_checkpoint(
      self,
      manual: bool = False,
      path: str = None
    ) -> str:
    if manual:
      self.logger.warn(
        "Manual checkpoints are not implemented for MORL algorithms"
      )
    checkpoint_dir = path if path is not None else os.path.join(
      self.logdir, f"checkpoints/{self.last_iteration()}"
    )
    os.makedirs(checkpoint_dir, exist_ok = True)
    self.algo.save(
      save_replay_buffer = self.additional_params.get(
        "save_replay_buffer", True
      ),
      save_dir = checkpoint_dir, 
      filename = self.generator.algo
    )
    last_checkpoint = os.path.join(
      checkpoint_dir, f"{self.generator.algo}.tar"
    )
    self.logger.log(
      "an Algorithm checkpoint has been created inside directory: "
      f"'{last_checkpoint}'",
      1
    )
    return last_checkpoint

  def load_checkpoint(self, path: str):
    """
    Load the provided `Algorithm` checkpoint
    """
    if not os.path.exists(path):
      raise FileNotFoundError(
        f"ERROR: checkpoint path {path} does not exist"
      )
    self.algo.load(
      path, 
      load_replay_buffer = self.additional_params.get(
        "load_replay_buffer", True
      )
    )
    # restore information concerning the last executed iteration
    self.iteration = int(os.path.basename(os.path.split(path)[0])) + 1
  
  def update_train_results(self, records):
    results = {
      "training_iteration": self.last_iteration(),
      "timesteps_total": self.algo.global_step,
      "wandb_logged_metrics": records
    }
    # convert to dataframe
    df = pd.DataFrame(results["wandb_logged_metrics"])
    if len(df) > 0:
      df["global_step"] = df["global_step"].ffill()
      df = df.groupby("global_step").first().reset_index()
      df["training_iteration"] = results["training_iteration"]
      df["timesteps_total"] = results["timesteps_total"]
      # print
      # -- load previous results (if any)
      results_file_path = os.path.join(self.logdir, 'progress.csv')
      if os.path.exists(results_file_path):
        previous_results = pd.read_csv(results_file_path)
        df = pd.concat([previous_results, df], ignore_index = True)
      # -- write
      df.to_csv(results_file_path, index = False)
    return results

class WandbMetricsCollector:
  def __init__(self):
    self.records = []
    self._original_log = None

  def __enter__(self):
    self._original_log = wandb.log
    def log(data, *args, **kwargs):
      self.records.append(dict(data))
      return self._original_log(data, *args, **kwargs)
    wandb.log = log
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    wandb.log = self._original_log
    return False
