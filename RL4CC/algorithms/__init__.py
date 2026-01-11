import RL4CC.log_and_report
import RL4CC.environment
import RL4CC.models

from RL4CC.algorithms.generators.mappo import MAPPOConfig

from ray.rllib.algorithms.registry import (
  ALGORITHMS_CLASS_TO_NAME,
  POLICIES,
  ALGORITHMS
)

def _import_mappo():
  import RL4CC.algorithms.generators.mappo as mappo
  return mappo.MAPPO, mappo.MAPPO.get_default_config()

ALGORITHMS_CLASS_TO_NAME["MAPPO"] = "MAPPO"
ALGORITHMS["MAPPO"] = _import_mappo
POLICIES["CCPPOTorchPolicy"] = RL4CC.models.centralized_critic_model.CCPPOTorchPolicy
