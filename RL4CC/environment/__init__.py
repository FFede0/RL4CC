from RL4CC.environment.base_environment import BaseEnvironment

from ray.tune.registry import register_env

register_env("BaseEnvironment", lambda config: BaseEnvironment(config))