from RL4CC.environment.base_environment import BaseEnvironment
from RL4CC.environment.base_multiagent_environment import BaseMultiAgentEnvironment

from ray.tune.registry import register_env

register_env("BaseEnvironment", lambda config: BaseEnvironment(config))
register_env("BaseMultiAgentEnvironment", lambda config: BaseMultiAgentEnvironment(config))