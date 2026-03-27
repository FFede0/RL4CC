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
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.env_context import EnvContext
from gymnasium.spaces import Discrete, Box, Dict
import numpy as np


class BaseMultiAgentEnvironment(MultiAgentEnv):
  """
  Base multi-agent environment dealing only with the simulation time 
  management. All agents are assumed to act simultaneously in each time step
  """
  def __init__(self, env_config: EnvContext):
    seed = self.load_configuration(env_config)
    # observation space
    self.define_observation_spaces()
    # define action space
    self.define_action_spaces()
    # reset
    self.reset(seed=seed)
    self._action_space_in_preferred_format = True
    self._obs_space_in_preferred_format = True
  
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
    # list of agent names (example: agent_0,...,agent_A)
    self.agents = env_config["agents"]
    return seed
  
  def define_observation_spaces(self):
    """
    Define the environment observation space(s)
    """
    self._observation_space = Dict({
      agent: Box(
        low = self.min_time, 
        high = self.max_time, 
        shape = (1,)
      ) for agent in self.agents
    })
  
  @property
  def observation_space(self):
    return self._observation_space
  
  def define_action_spaces(self):
    """
    Define the environment action space(s)
    """
    # {do nothing}
    self._action_space = Dict({
      agent: Discrete(1) for agent in self.agents
    })
  
  @property
  def action_space(self):
    return self._action_space
  
  def observation(self):
    """
    Return the next observation (and the corresponding info dictionary)
    ---
    In general, the returned observations dict must contain those agents 
    (and only those agents) that should act next. Agent IDs that should NOT 
    act in the next step() call must NOT have their observations in the 
    returned observations dict
    """
    obs = {
      agent: np.array([self.current_time]) for agent in self.agents
    }
    reward = self.compute_reward()
    obs_info = {
      "__common__": {
        "current_time": self.current_time
      },
      **{
        agent: {"reward": r} for agent, r in reward.items()
      }
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
  
  def step(self, action_dict):
    """
    Applies the action chosen by each agent, moves to the next state and 
    computes the reward
    ---
    Returns a tuple containing:
      1) new observations for each ready agent, 
      2) reward values for each ready agent. If the episode is just started, 
      the value will be None. 
      3) Terminated values for each ready agent. The special key “__all__” 
      (required) is used to indicate env termination. 
      4) Truncated values for each ready agent. 
      5) Info values for each agent id (may be empty dicts)
    """
    # compute reward
    reward = self.compute_reward()
    # update time
    self.current_time += self.time_step
    # check if we are in the last step of the episod should be truncated
    done = {
      agent: self.current_time >= self.max_time for agent in list(self.agents) + [
        "__all__"
      ]
    }
    truncated = done
    # define observation
    obs, obs_info = self.observation()
    return obs, reward, done, truncated, obs_info

  def compute_reward(self):
    # return 1.0 in the last step, 0.0 otherwise
    reward = 0.0
    if self.current_time + self.time_step >= self.max_time:
      reward = 1.0
    return {agent: reward for agent in self.agents}
