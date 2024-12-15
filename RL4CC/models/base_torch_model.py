"""
Copyright 2024 Mohanad Diab, Federica Filippini

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

from RL4CC.utilities.logger import Logger

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict

from abc import ABC, abstractmethod
from gymnasium.spaces import Space
from torch import nn


class BaseTorchModel(TorchModelV2, nn.Module, ABC):
  def __init__(
      self,
      obs_space: Space,
      action_space: Space,
      num_outputs: int,
      model_config: ModelConfigDict,
      name: str,
      **kwargs
    ):
    super().__init__(obs_space, action_space, num_outputs, model_config, name)
    nn.Module.__init__(self)
    self.logger = Logger("RL4CC-Torch Model Logger")

  @abstractmethod
  def forward(self, input_dict, state, seq_lens, **kwargs):
    """
    Implementation of the forward method logic

    return: logits, state
    """
    pass

  @abstractmethod
  def value_function(self):
    """
    Implementation of the forward method logic

    return: value function
    """
    pass
