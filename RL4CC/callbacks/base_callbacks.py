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
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv

from typing import Dict, Tuple
import numpy as np


class BaseCallbacks(DefaultCallbacks):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.RELEVANT_KEYS = [
      "current_time",
      "reward"
    ]
  
  def on_episode_start(
      self,
      *,
      worker: RolloutWorker,
      base_env: BaseEnv,
      policies: Dict[str, Policy],
      episode: Episode,
      env_index: int,
      **kwargs,
    ):
    # # make sure this episode has just been started (only initial obs
    # # logged so far).
    # assert episode.length == 0, (
    #   "ERROR: `on_episode_start()` callback should be called right "
    #   "after env reset!"
    # )
    # create lists to store info (in user_data and hist_data)
    for key in self.RELEVANT_KEYS:
      episode.user_data[key] = []
      episode.hist_data[key] = []
    # add worker index
    episode.user_data["worker_index"] = []
    episode.hist_data["worker_index"] = []
  
  def on_episode_step(
      self,
      *,
      worker: RolloutWorker,
      base_env: BaseEnv,
      policies: Dict[str, Policy],
      episode: Episode,
      env_index: int,
      **kwargs,
    ):
    # # make sure this episode is ongoing
    # assert episode.length > 0, (
    #   "ERROR: `on_episode_step()` callback should not be called right "
    #   "after env reset!"
    # )
    for key in self.RELEVANT_KEYS:
      val = episode.last_info_for()[key]
      if isinstance(val, np.ndarray):
        val = val.tolist()
      # add to user_data
      episode.user_data[key].append(val)
    # add worker index
    episode.user_data["worker_index"].append(worker.worker_index)
  
  def on_episode_end(
      self,
      *,
      worker: RolloutWorker,
      base_env: BaseEnv,
      policies: Dict[str, Policy],
      episode: Episode,
      env_index: int,
      **kwargs,
    ):
    # # Make sure this episode is really done.
    # assert episode.batch_builder.policy_collectors["default_policy"].batches[
    #   -1
    # ]["dones"][-1], (
    #   "ERROR: `on_episode_end()` should only be called "
    #   "after episode is done!"
    # )
    # add to hist data and add averages to custom metrics
    for key in self.RELEVANT_KEYS:
      episode.hist_data[key] = episode.user_data[key]
      episode.custom_metrics[f"{key}_avg"] = np.mean(episode.user_data[key])
    # add worker index
    episode.hist_data["worker_index"] = episode.user_data["worker_index"]
  
  def on_sample_end(
      self, 
      *, 
      worker: RolloutWorker, 
      samples: SampleBatch, 
      **kwargs
    ):
    pass

  def on_train_result(self, *, algorithm, result: dict, **kwargs):
    # you can mutate the result dict to add new fields to return
    result["callback_ok"] = True
    # normally, RLlib would aggregate any custom metric into a mean, max and 
    # min of the given metric
    if "custom_metrics" in result:
      keys = list(result["custom_metrics"].keys())
      vals = list(result["custom_metrics"].values())
      for key, val in zip(keys, vals):
        result["custom_metrics"][f"{key}_avg"] = np.mean(val)
        result["custom_metrics"][f"{key}_min"] = np.min(val)
        result["custom_metrics"][f"{key}_max"] = np.max(val)
  
  def on_learn_on_batch(
      self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
    result["sum_actions_in_train_batch"] = train_batch["actions"].sum()
  
  def on_postprocess_trajectory(
      self,
      *,
      worker: RolloutWorker,
      episode: Episode,
      agent_id: str,
      policy_id: str,
      policies: Dict[str, Policy],
      postprocessed_batch: SampleBatch,
      original_batches: Dict[str, Tuple[Policy, SampleBatch]],
      **kwargs,
    ):
    pass