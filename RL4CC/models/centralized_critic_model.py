try:
  from ray.rllib.examples._old_api_stack.models.centralized_critic_models import (
    TorchCentralizedCriticModel
  )
except ModuleNotFoundError:
  from ray.rllib.examples.models.centralized_critic_models import (
    TorchCentralizedCriticModel
  )
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.models.torch.misc import SlimFC
from gymnasium.spaces import Discrete

from gymnasium.spaces.utils import flatdim

torch, nn = try_import_torch()
import numpy as np


class CentralizedValueMixin:
  """Add method to evaluate the central value function from the model."""
  def __init__(self):
    self.compute_central_vf = self.model.central_value_function


class CCPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
  def __init__(self, observation_space, action_space, config):
    self.OPPONENT_OBS = "opponent_obs"
    self.OPPONENT_ACTION = "opponent_action"
    PPOTorchPolicy.__init__(self, observation_space, action_space, config)
    CentralizedValueMixin.__init__(self)

  def _centralized_critic_postprocessing(
      self, sample_batch, other_agent_batches = None, episode = None
    ):
    """
    Grabs the opponent obs/act and includes it in the experience train_batch,
    and computes GAE using the central vf predictions.
    """
    # get model configuration info
    custom_model_config = self.config["model"].get("custom_model_config", {})
    mode = custom_model_config.get("mode", "concat")
    # run postprocessing
    if hasattr(self, "compute_central_vf"):
      assert other_agent_batches is not None
      if self.config["enable_connectors"]:
        # values() -> list of (other_policy, _, batch)
        opponent_batches = [
          triple[-1] for triple in other_agent_batches.values()
        ]
      else:
        # values() -> list of (other_policy, batch)
        opponent_batches = [pair[-1] for pair in other_agent_batches.values()]
      # separate lists of observations and actions
      op_obs_list = [b[SampleBatch.CUR_OBS] for b in opponent_batches]
      op_act_list = []
      for b in opponent_batches:
        original_shape = b[SampleBatch.ACTIONS].shape
        arr2d = np.atleast_2d(b[SampleBatch.ACTIONS])
        if original_shape != arr2d.shape:
          arr2d = arr2d.T
        op_act_list.append(arr2d)
      # -- define observations and actions
      opponent_obs, opponent_actions = None, None
      if mode == "avg":
        # --- average
        opponent_obs = np.mean(np.stack(op_obs_list, axis=0), axis=0)
        opponent_actions = np.mean(np.stack(op_act_list, axis=0), axis=0)
      elif mode == "concat":
        # --- concatenate along feature dimension
        opponent_obs = np.concatenate(op_obs_list, axis = 1)
        opponent_actions = np.concatenate(op_act_list, axis = 1)
      # record the opponent obs and actions in the trajectory
      sample_batch[self.OPPONENT_OBS] = opponent_obs
      sample_batch[self.OPPONENT_ACTION] = opponent_actions
      # overwrite default VF prediction with the central VF
      sample_batch[SampleBatch.VF_PREDS] = (
        self.compute_central_vf(
          convert_to_torch_tensor(
            sample_batch[SampleBatch.CUR_OBS], self.device
          ),
          convert_to_torch_tensor(
            sample_batch[self.OPPONENT_OBS], self.device
          ),
          convert_to_torch_tensor(
            sample_batch[self.OPPONENT_ACTION], self.device
          ),
        ).cpu().detach().numpy()
      )
    else:
      n_other_agents = custom_model_config.get("n_agents", 2) - 1
      if mode == "avg":
        n_other_agents = 1
      # policy hasn't been initialized yet, use zeros.
      sample_batch[self.OPPONENT_OBS] = np.zeros_like(
        np.concatenate(
          [
            sample_batch[SampleBatch.CUR_OBS] for _ in range(n_other_agents)
          ],
          axis = 1
        )
      )
      original_shape = sample_batch[SampleBatch.ACTIONS].shape
      sample_act = np.atleast_2d(sample_batch[SampleBatch.ACTIONS])
      if original_shape != sample_act.shape:
        sample_act = sample_act.T
      sample_batch[self.OPPONENT_ACTION] = np.zeros_like(
        np.concatenate(
          [
            np.atleast_2d(sample_act) for _ in range(n_other_agents)
          ],
          axis = 1
        )
      )
      sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(
        sample_batch[SampleBatch.REWARDS], dtype=np.float32
      )
    # record if completed
    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
      last_r = 0.0
    else:
      last_r = sample_batch[SampleBatch.VF_PREDS][-1]
    # compute advantage
    train_batch = compute_advantages(
      sample_batch,
      last_r,
      self.config["gamma"],
      self.config["lambda"],
      use_gae = self.config["use_gae"]
    )
    return train_batch

  def _loss_with_central_critic(self, model, dist_class, train_batch):
    # save original value function
    vf_saved = model.value_function
    # calculate loss with a custom value function
    model.value_function = lambda: self.model.central_value_function(
      train_batch[SampleBatch.CUR_OBS],
      train_batch[self.OPPONENT_OBS],
      train_batch[self.OPPONENT_ACTION],
    )
    self._central_value_out = model.value_function()
    loss = super().loss(model, dist_class, train_batch)
    # restore original value function
    model.value_function = vf_saved
    return loss

  def loss(self, model, dist_class, train_batch):
    return self._loss_with_central_critic(model, dist_class, train_batch)

  def postprocess_trajectory(
      self, sample_batch, other_agent_batches = None, episode = None
    ):
    return self._centralized_critic_postprocessing(
      sample_batch, other_agent_batches, episode
    )


class CustomTorchCCModel(TorchCentralizedCriticModel):
  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    super().__init__(
      obs_space, action_space, num_outputs, model_config, name
    )
    # get model configuration info
    custom_model_config = model_config.get("custom_model_config", {})
    n_agents = custom_model_config["n_agents"]
    fcnet_hiddens = custom_model_config.get("central_vf_fcnet_hiddens", [16])
    fcnet_act = custom_model_config.get("central_vf_fcnet_activation", "tanh")
    mode = custom_model_config.get("mode", "concat")
    n_obs, n_acts = None, None
    if mode == "concat":
      n_obs = n_agents      # obs + opp_obs for all agents
      n_acts = n_agents - 1 # opp_act
    elif mode == "avg":
      n_obs = 2             # obs + avg of opp_obs
      n_acts = 1            # avg of opp_act
    # central VF maps (obs, opp_obs, opp_act) -> vf_pred
    input_size = None
    try:
      #input_size = flatdim(obs_space) * n_obs + flatdim(action_space) * n_act
      act_dim = 1 if isinstance(action_space, Discrete) else flatdim(action_space)
      input_size = flatdim(obs_space) * n_obs + act_dim * n_acts
    except Exception:
      input_size = flatdim(obs_space) * n_obs + action_space.dim * n_acts
    if len(fcnet_hiddens) > 1:
      hiddens = []
      prev_size = input_size
      for n in fcnet_hiddens[:-1]:
        hiddens.append(SlimFC(prev_size, n, activation_fn=fcnet_act))
        prev_size = n
      self.central_vf = nn.Sequential(
        *hiddens,
        SlimFC(fcnet_hiddens[-1], 1),
      )
    else:
      self.central_vf = nn.Sequential(
        SlimFC(input_size, fcnet_hiddens[0], activation_fn=fcnet_act),
        SlimFC(fcnet_hiddens[-1], 1),
      )

  def forward(self, input_dict, state, seq_lens):
    model_out, _ = self.model.forward(input_dict, state, seq_lens)
    return model_out, []

  def central_value_function(self, obs, opponent_obs, opponent_actions):
    # obs:             [B, obs_dim]
    # opponent_obs:    [B, N*obs_dim] or [B, obs_dim] (if averaged)
    # opponent_actions:[B, N*act_dim] or [B, act_dim]
    input_ = torch.cat(
      [
        obs,
        opponent_obs,
        opponent_actions,
      ],
      1,
    ).float()
    return torch.reshape(self.central_vf(input_), [-1])
