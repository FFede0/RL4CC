import RL4CC.log_and_report
import RL4CC.environment
import RL4CC.models

from RL4CC.algorithms.mappo import MAPPOConfig
from RL4CC.algorithms.vdn_qmix import VDNConfig, QMIXConfig, ValueDecompPolicy

from ray.rllib.algorithms.registry import (
  ALGORITHMS_CLASS_TO_NAME,
  POLICIES,
  ALGORITHMS
)

def _import_mappo():
  import RL4CC.algorithms.mappo as mappo
  return mappo.MAPPO, mappo.MAPPO.get_default_config()

def _import_vdn():
  import RL4CC.algorithms.vdn_qmix as vdn_qmix
  return vdn_qmix.VDN, vdn_qmix.VDN.get_default_config()

def _import_qmix():
  import RL4CC.algorithms.vdn_qmix as vdn_qmix
  return vdn_qmix.QMIX, vdn_qmix.QMIX.get_default_config()

ALGORITHMS_CLASS_TO_NAME["MAPPO"] = "MAPPO"
ALGORITHMS["MAPPO"] = _import_mappo
ALGORITHMS_CLASS_TO_NAME["VDN"] = "VDN"
ALGORITHMS["VDN"] = _import_vdn
ALGORITHMS_CLASS_TO_NAME["QMIX"] = "QMIX"
ALGORITHMS["QMIX"] = _import_qmix
POLICIES["CCPPOTorchPolicy"] = RL4CC.models.centralized_critic_model.CCPPOTorchPolicy
POLICIES["ValueDecompPolicy"] = ValueDecompPolicy
