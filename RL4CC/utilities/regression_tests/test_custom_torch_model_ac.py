import RL4CC.models  # registers the custom models

from ray.rllib.models import MODEL_DEFAULTS, ModelCatalog
from gymnasium.spaces import Box, Discrete
from torch import nn
import numpy as np
import torch


def build_model(vf_hiddens: list = None):
  custom_model_config = {"n_features": [64, 32], "fun_layers": ["ReLU", "ReLU"]}
  if vf_hiddens is not None:
    custom_model_config["vf_hiddens"] = vf_hiddens
  return ModelCatalog.get_model_v2(
    obs_space = Box(-1.0, 1.0, (10,), np.float32),
    action_space = Discrete(5),
    num_outputs = 5,
    model_config = {
      **MODEL_DEFAULTS,
      "custom_model": "custom_torch_model_ac",
      "custom_model_config": custom_model_config
    },
    framework = "torch"
  )


def linear_sizes(network: nn.Module) -> list:
  return [m.out_features for m in network.modules() if isinstance(m, nn.Linear)]


def main():
  torch.manual_seed(0)
  model = build_model()
  # critic layers: default and user-defined
  assert linear_sizes(model.critic) == [512, 256, 1]
  assert linear_sizes(build_model([16]).critic) == [16, 1]
  # output shapes
  logits, _ = model({"obs": torch.rand(8, 10) * 2 - 1})
  value = model.value_function()
  assert logits.shape == (8, 5), logits.shape
  assert value.shape == (8,), value.shape
  # V(s) comes from the critic, not from the maximum actor logit
  assert not torch.allclose(value, logits.max(dim = 1).values)
  # the value loss reaches the critic and leaves the actor untouched
  value.pow(2).mean().backward()
  assert all(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in model.critic.parameters()
  )
  assert all(p.grad is None for p in model.network.parameters())
  print("test_custom_torch_model_ac: all checks passed")


if __name__ == "__main__":
  main()
