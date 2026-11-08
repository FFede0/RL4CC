from RL4CC.models.custom_torch_model import CustomTorchModel

from ray.rllib.models.torch.misc import SlimFC
from torch import nn
import torch


class CustomTorchModelAC(CustomTorchModel):
  """
  Actor-critic version of CustomTorchModel.

  The actor is exactly the CustomTorchModel network, while V(s) is computed
  by a separate critic network that shares no weights with the actor:
  obs -> vf_hiddens (ReLU) -> 1. The critic hidden layers are set through
  the `vf_hiddens` key of `custom_model_config` (default: [512, 256]).
  """
  def __init__(
      self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
    super().__init__(obs_space, action_space, num_outputs, model_config, name)
    vf_hiddens = self.model_config.get("custom_model_config", {}).get(
      "vf_hiddens", [512, 256]
    )
    prev, layers = self.n_input, []
    for n in vf_hiddens:
      layers.append(SlimFC(prev, n, activation_fn="relu"))
      prev = n
    layers.append(SlimFC(prev, 1))
    self.critic = nn.Sequential(*layers)
    self._value = None

  def forward(self, input_dict, state, seq_lens, **kwargs):
    """
    Compute the actor logits and store the critic output

    return: logits, state
    """
    logits, state = super().forward(input_dict, state, seq_lens, **kwargs)
    self._value = torch.reshape(
      self.critic(input_dict["obs_flat"].float()), [-1]
    )
    return logits, state

  def value_function(self):
    """
    return: value function computed by the critic network
    """
    return self._value
