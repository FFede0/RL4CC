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
from RL4CC.models.base_tf_model import BaseTFModel

from ray.rllib.utils.typing import ModelConfigDict
from gymnasium.spaces import Space
import tensorflow as tf
import inspect


class CustomTFModel(BaseTFModel):
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
    config = model_config.get("custom_model_config", {})
    # define input and output shapes
    self.n_input = obs_space.shape[0]
    self.n_output = action_space.n
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
        layers = [item for item in inspect.getmembers(tf.keras.layers, inspect.isclass) if fun_layer in item]
        if len(layers) > 0:
          layer = layers[0][1]
          hidden_layers.append(layer())
        else:
          raise NotImplementedError(
            f"function {fun_layer} is not defined in tf.keras.layers"
          )
        if dropout and dropout_list[idx] > 0:
          hidden_layers.append(tf.keras.layers.Dropout(dropout_list[idx]))
        if idx < len(n_features) - 1 and n_features[idx] != n_features[idx + 1]:
          hidden_layers.append(tf.keras.layers.Dense(n_features[idx + 1]))
        idx += 1
    else:
      raise RuntimeError(
        "n_features and fun_layers should be lists with the same length"
      )
    # define layers
    _L1 = tf.keras.layers.Dense(
      n_features[0], 
      input_shape = (self.n_input,), 
      activation = 'relu',
      kernel_initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    )
    _L2 = tf.keras.layers.Dense(
      self.n_output, 
      activation = 'relu',
      kernel_initializer = tf.keras.initializers.GlorotUniform(seed=seed)
    )
    # define network
    self.network = tf.keras.Sequential([_L1] + hidden_layers + [_L2])

  def forward(self, input_dict, state, seq_lens, **kwargs):
    """
    Implementation of the forward method logic

    return: logits, state
    """
    obs = tf.cast(input_dict["obs"], tf.float32)
    q_values = self.network(obs)
    self._last_q_values = q_values
    return q_values, state

  def value_function(self):
    """
    Implementation of the forward method logic

    return: value function
    """
    return tf.reduce_max(self._last_q_values, axis=1)
