from .base_tf_model import BaseTFModel
from .base_torch_model import BaseTorchModel
from .custom_tf_model import CustomTFModel
from .custom_torch_model import CustomTorchModel
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("custom_tf_model", CustomTFModel)
ModelCatalog.register_custom_model("custom_torch_model", CustomTorchModel)

