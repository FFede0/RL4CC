from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from RL4CC.models.centralized_critic_model import CCPPOTorchPolicy

from ray.rllib.algorithms.ppo import PPO, PPOConfig


class MAPPO(PPO):
  @classmethod
  def get_default_policy_class(cls, config):
    return CCPPOTorchPolicy
  
  @classmethod
  def get_default_config(cls) -> AlgorithmConfig:
    return MAPPOConfig()


class MAPPOConfig(PPOConfig):
  def __init__(self, algo_class = None):
    super().__init__(algo_class = algo_class or MAPPO)
