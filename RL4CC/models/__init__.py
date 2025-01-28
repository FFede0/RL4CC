from ray.rllib.models import ModelCatalog

from RL4CC.utilities.logger import Logger

logger = Logger(name="RL4CC-Models")

# Try loading the custom models. The user can only have one framework (Torch or
# Tensorflow) installed.
try:
  from .base_torch_model import BaseTorchModel
  from .custom_torch_model import CustomTorchModel
  ModelCatalog.register_custom_model("custom_torch_model", CustomTorchModel)
except ModuleNotFoundError as error:
  logger.warn(f"Could not register CustomTorchModel: No module named {error.name!r}.")

try:
  from .base_tf_model import BaseTFModel
  from .custom_tf_model import CustomTFModel
  ModelCatalog.register_custom_model("custom_tf_model", CustomTFModel)
except ModuleNotFoundError as error:
  logger.warn(f"Could not register CustomTFModel: No module named {error.name!r}.")
