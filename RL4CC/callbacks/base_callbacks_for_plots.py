"""
Copyright 2024 Riccardo Cavadini, Federica Filippini

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
from RL4CC.callbacks.base_callbacks import BaseCallbacks

from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv

import os
import json
import numpy as np
from typing import Dict, Tuple

class BaseCallbacksForPlots(BaseCallbacks):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.RELEVANT_KEYS = [
      "current_time"
    ]
    self.iteration_id = 0
    self.step_id = 0

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
    for key in self.RELEVANT_KEYS:
      episode.user_data[key] = []
      episode.hist_data[key] = []
      episode.custom_metrics[key] = []
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
    # make sure this episode is ongoing
    assert episode.length > 0, (
      "ERROR: `on_episode_step()` callback should not be called right "
      "after env reset!"
    )
    for key in self.RELEVANT_KEYS:
      val = episode.last_info_for()[key]
      if not isinstance(val, (int, float, list)):
        if isinstance(val, np.ndarray):
          val = val.tolist()
        elif isinstance(val, np.int32) or isinstance(val, np.int64):
          val = int(val)
        elif isinstance(val, np.float32) or isinstance(val, np.float64):
          val = float(val)
        elif np.isnan(val):
          val = None
      # add to user_data
      episode.user_data[key].append(val)
      # add worker index
    episode.user_data["worker_index"].append(worker.worker_index) 
    self.step_id += 1 
  
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
    # check if there are multiple episodes in a batch, i.e.
    # "batch_mode": "truncate_episodes".
    if worker.config.batch_mode == "truncate_episodes":
      # Make sure this episode is really done.
      if hasattr(episode, "batch_builder"):
        assert episode.batch_builder.policy_collectors["default_policy"].batches[
          -1
        ]["dones"][-1], (
          "ERROR: `on_episode_end()` should only be called "
          "after episode is done!"
        )
    # # add averages to custom metrics
    # response_time = np.mean(episode.user_data["response_times"])
    # episode.custom_metrics["response_times_avg"] = response_time
    # add to hist data
    for key in self.RELEVANT_KEYS:
      if key in episode.user_data:
        episode.hist_data[key] = episode.user_data[key]
        episode.custom_metrics[key] = episode.user_data[key]  
    # add worker index
    episode.hist_data["worker_index"] = episode.user_data["worker_index"] 
  
  def on_sample_end(
      self, 
      *, 
      worker: RolloutWorker, 
      samples: SampleBatch, 
      **kwargs
    ):
    # TODO: any sanity check to perform?
    pass

  def on_train_result(self, *, algorithm, result: dict, **kwargs):
    custom_metrics = []
    if (
      "env_runners" in result.keys() and 
        "custom_metrics" in result["env_runners"].keys()
    ):
      custom_metrics = result["env_runners"]["custom_metrics"]
    elif (
        "custom_metrics" in result.keys() and 
          len(result["custom_metrics"].keys()) > 0
    ):
      custom_metrics = result["custom_metrics"]
    else:
      result["callback_ok"] = False
      return    
    first_key = list(custom_metrics.keys())[0]
    random_index = np.random.randint(0, len(custom_metrics[first_key]))   
    random_custom_metrics = {}
    for key in self.RELEVANT_KEYS:
      if key in custom_metrics:
        random_custom_metrics[key] = custom_metrics[key][random_index]  
    simulation_folder = algorithm.config["logger_config"]["logdir"] 
    it_id = self.iteration_id 
    os.makedirs(f"{simulation_folder}/custom_metrics", exist_ok = True)
    filename = f"{simulation_folder}/custom_metrics/iteration_{it_id}.json" 
    with open(filename, "w+") as f:
      f.write(json.dumps(random_custom_metrics, indent=2))    
    self.iteration_id += 1  
    # you can mutate the result dict to add new fields to return
    result["callback_ok"] = True  
  
  def on_learn_on_batch(
      self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    ) -> None:
    result["sum_actions_in_train_batch"] = train_batch["actions"].sum()
    # # Log the sum of actions in the train batch.
    # print(
    #     "policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #         policy, result["sum_actions_in_train_batch"]
    #     )
    # ) 
  
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
