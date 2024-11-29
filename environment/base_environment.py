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
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Discrete, Box
import gymnasium as gym
import numpy as np


class BaseEnvironment(gym.Env):
  """
  Base environment dealing only with the simulation time management
  """
  def __init__(self, env_config: EnvContext):
    seed = self.load_configuration(env_config)
    # observation space
    self.define_observation_space()
    # define action space
    self.define_action_space()
    # reset
    self.reset(seed=seed)
  
  def load_configuration(self, env_config: EnvContext) -> int:
    """
    Initialize environment loading info from the provided configuration dict
    """
    # simulation time management
    self.min_time = env_config["min_time"]
    self.max_time = env_config["max_time"]
    self.time_step = env_config["time_step"]
    self.current_time = self.min_time
    # seed for randomization (None)
    seed = None
    return seed
  
  def define_observation_space(self):
    """
    Define the environment observation space
    """
    self.observation_space = Box(
      low = self.min_time, 
      high = self.max_time, 
      shape = (1,)
    )
  
  def define_action_space(self):
    """
    Define the environment action space
    """
    # {do nothing}
    self.action_space = Discrete(1)
  
  def observation(self):
    obs = np.array([self.current_time])
    obs_info = {
      "current_time": self.current_time,
      "reward": self.compute_reward()
    }
    return obs, obs_info

  def reset(self, seed: int = None, options = None):
    # set seed from the parent class
    super().reset(seed=seed)
    # restart time
    self.current_time = self.min_time
    # define observation
    obs, obs_info = self.observation()
    return obs, obs_info
  
  def step(self, action):
    # compute reward
    reward = self.compute_reward()
    # update time
    self.current_time += self.time_step
    # check if we are in the last step of the episod should be truncated
    done = self.current_time >= self.max_time
    truncated = done
    # define observation
    obs, obs_info = self.observation()
    return obs, reward, done, truncated, obs_info

  def compute_reward(self):
    # return 1.0 in the last step, 0.0 otherwise
    reward = 0.0
    if self.current_time + self.time_step >= self.max_time:
      reward = 1.0
    return reward
