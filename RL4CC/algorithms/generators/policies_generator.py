"""
Copyright 2025 Federica Filippini

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
from ray.rllib.policy.policy import PolicySpec


class PoliciesGenerator:
  def __init__(self) -> None:
    pass

  def generate_multiagent_config(
      self, agents: list, policy_config: dict
    ) -> dict:
    # each agent has its own policy (policy_node_X in policy -> node_X in the 
    # environment).
    multiagent_config = {
      "policies": {},
      "policies_to_train": []
    }
    for agent in agents:
      policy_name = agent
      agent_config = None
      if policy_config is not None:
        agent_config = policy_config.get(agent, None)
      multiagent_config["policies"][policy_name] = PolicySpec(
        policy_class = None,  # inferred from Algorithm
        observation_space = None, # inferred from the environment
        action_space = None, # inferred from the environment
        config = agent_config,
      )
      # -- initially, all policies should be trained
      multiagent_config["policies_to_train"].append(policy_name)
    # function mapping agent to policy
    multiagent_config["policy_mapping_fn"] = self.policy_mapping_fn
    return multiagent_config
  
  @staticmethod
  def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    """
    Called by RLlib at each step to map an agent to a policy.
    In this case, the map is static: every agent has the same policy, and a
    policy has the same single agent.
    """
    return agent_id
      