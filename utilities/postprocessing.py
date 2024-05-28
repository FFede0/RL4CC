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
from environment.base_environment import BaseEnvironment

from ray.rllib.policy.policy import Policy


def evaluate_policy(
    policy: Policy, 
    env: BaseEnvironment,
    num_eval_episodes: int = 1,
    explore: bool = False
  ) -> list:
  """
  Evaluate `policy` on the given environment, for a specified number of 
  episodes
  ---
  Returns
    a list of observations, actions and related metrics collected during 
    evaluation
    Structure:
    [
      {
        "episode": int,
        "evaluation_steps": dict,
        "total_reward": float
      }
    ] 

    Structure of `evaluation_steps`:
    [
      {
        "step": int,
        "state": np.ndarray
        "action": tuple
        "next_state":
        "reward": float
        "done": bool
        "truncated": bool
        "obs_info": dict
        "total_reward": float
      }
    ]
  """
  evaluation_episodes = []
  # start evaluation
  episode = 0
  total_reward = 0.0
  while episode < num_eval_episodes:
    evaluation_steps = []
    # reset environment
    obs, _ = env.reset()
    done = False
    total_episode_reward = 0.0
    step = 0
    # run episode
    while not done:
      action = policy.compute_single_action(obs, explore=explore)
      next_obs, reward, done, truncated, obs_info = env.step(action)
      total_episode_reward += reward
      evaluation_steps.append({
        "step": step,
        "state": obs,
        "action": action,
        "next_state": next_obs,
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "obs_info": obs_info,
        "total_reward": total_episode_reward
      })
      # move to next state
      obs = next_obs
      step += 1
    # move to next episode
    total_reward += total_episode_reward
    evaluation_episodes.append({
      "episode": episode,
      "evaluation_steps": evaluation_steps,
      "total_reward": total_reward
    })
    episode += 1
  return evaluation_episodes
