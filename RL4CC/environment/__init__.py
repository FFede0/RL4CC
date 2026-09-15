
# Ray-based single- and multi-agent base environments
try:
  from RL4CC.environment.base_environment import BaseEnvironment
  from RL4CC.environment.base_multiagent_environment import BaseMultiAgentEnvironment
  from ray.tune.registry import register_env
  register_env("BaseEnvironment", lambda config: BaseEnvironment(config))
  register_env("BaseMultiAgentEnvironment", lambda config: BaseMultiAgentEnvironment(config))
except ImportError:
  pass

# MO-Gymnasium-based multi-objective base environment
from gymnasium.envs.registration import register
register(
  id = "BaseMultiObjectiveEnvironment",
  entry_point = "RL4CC.environment.base_multiobjective_environment:BaseMultiObjectiveEnvironment",
)
