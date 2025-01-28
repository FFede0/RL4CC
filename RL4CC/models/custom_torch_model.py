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

import collections
from RL4CC.models.base_torch_model import BaseTorchModel

from ray.rllib.utils.typing import ModelConfigDict

from gymnasium.spaces import Space
from torch import nn
import inspect
import torch


class CustomTorchModel(BaseTorchModel):
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
    self._last_q_values = None
    # fetch the custom model config
    config = self.model_config.get("custom_model_config", {})
    # define input and output shapes
    self.n_input = self.obs_space.shape[0]
    self.n_output = self.action_space.n
    self.logger.log("input shape: " + str(obs_space.shape), 2)
    self.logger.log("output shape: " + str(action_space.n), 2)
    # fetch the model parameters from the config dict
    n_features = config.get("n_features", [])
    fun_layers = config.get("fun_layers", [])
    dropout = config.get("dropout", False)
    dropout_list = config.get("dropout_list", [])
    seed = config.get("seed", 1234)
    # build structure
    if len(n_features) > 0 and len(n_features) == len(fun_layers):
      hidden_layers = []
      idx = 0
      for (n_feature, fun_layer) in zip(n_features, fun_layers):
        layers = [
          item for item in inspect.getmembers(
            nn, inspect.isclass
          ) if fun_layer in item
        ]
        if len(layers) > 0:
          layer = layers[0][1]
          hidden_layers.append(layer())
        else:
          raise NotImplementedError(
            f"function {fun_layer} is not defined in torch.nn"
          )
        if dropout and dropout_list[idx] > 0:
          hidden_layers.append(nn.Dropout(dropout_list[idx]))
        if idx < len(n_features) - 1 and n_features[idx] != n_features[idx + 1]:
          hidden_layers.append(nn.Linear(n_features[idx], n_features[idx + 1]))
        idx += 1
    else:
        raise RuntimeError(
          "n_features and fun_layers should be lists with the same length"
        )
    # define layers
    _L1 = nn.Linear(self.n_input, n_features[0])
    nn.init.xavier_uniform_(_L1.weight, gain=nn.init.calculate_gain('relu'))
    _L2 = nn.Linear(n_features[-1], self.n_output)
    nn.init.xavier_uniform_(_L2.weight, gain=nn.init.calculate_gain('relu'))
    # define network
    self.network = nn.Sequential(_L1)
    for layer in hidden_layers:
      self.network.append(layer)
    self.network.append(_L2)

  def forward(self, input_dict, state, seq_lens, **kwargs):
    """
    Implementation of the forward method logic

    return: logits, state
    """

    obs = input_dict["obs"]
    
    if isinstance(obs, torch.Tensor):
        obs = obs.float()
    elif isinstance(obs, collections.OrderedDict):
        # Handle the OrderedDict case
        tensors = []
        for key, value in obs.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Expected each item in input_dict['obs'] to be a torch.Tensor but got {type(value)} for key '{key}'")
            tensors.append(value.float())
        # Concatenate tensors along the last dimension (assuming they need to be concatenated)
        obs = torch.cat(tensors, dim=-1)
    else:
        raise TypeError(f"Expected input_dict['obs'] to be a torch.Tensor or OrderedDict but got {type(obs)}")

    q_values = self.network(obs)
    self._last_q_values = q_values
    return q_values, state

  def value_function(self):
    """
    Implementation of the forward method logic

    return: value function
    """
    return torch.max(self._last_q_values, dim=1)[0]
