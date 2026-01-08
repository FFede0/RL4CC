from RL4CC.models.centralized_critic_model import CCPPOTorchPolicy
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
POLICIES["CCPPOTorchPolicy"] = CCPPOTorchPolicy
