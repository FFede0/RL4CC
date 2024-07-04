import json
import os
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy import Policy
from ray.rllib.env import BaseEnv

from callbacks.base_callbacks import BaseCallbacks

from typing import Dict, Tuple
import numpy as np

class BaseCallbacksForPlots(BaseCallbacks):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.RELEVANT_KEYS = [
			"current_time"
		]

		self.iteration_id = 0
		self.step_id = 0

	def on_episode_start(
		self,
		*,
		worker: RolloutWorker,
		base_env: BaseEnv,
		policies: Dict[str, Policy],
		episode: Episode,
		env_index: int,
		**kwargs,
    ):

		for key in self.RELEVANT_KEYS:
			episode.user_data[key] = []
			episode.hist_data[key] = []
			episode.custom_metrics[key] = []
		# add worker index
		episode.user_data["worker_index"] = []
		episode.hist_data["worker_index"] = []

	def on_episode_step(
		self,
		*,
		worker: RolloutWorker,
		base_env: BaseEnv,
		policies: Dict[str, Policy],
		episode: Episode,
		env_index: int,
		**kwargs,
	):
     
		# make sure this episode is ongoing
		assert episode.length > 0, (
			"ERROR: `on_episode_step()` callback should not be called right "
			"after env reset!"
		)
		
		for key in self.RELEVANT_KEYS:
			val = episode.last_info_for()[key]
			if not isinstance(val, (int, float, list)):
				if isinstance(val, np.ndarray):
					val = val.tolist()
				elif isinstance(val, np.int32) or isinstance(val, np.int64):
					val = int(val)
				elif isinstance(val, np.float32) or isinstance(val, np.float64):
					val = float(val)
				elif np.isnan(val):
					val = None
			# add to user_data
			episode.user_data[key].append(val)
			# add worker index
		episode.user_data["worker_index"].append(worker.worker_index)

		self.step_id += 1

	def on_episode_end(
		self,
		*,
		worker: RolloutWorker,
		base_env: BaseEnv,
		policies: Dict[str, Policy],
		episode: Episode,
		env_index: int,
		**kwargs,
	):
		# check if there are multiple episodes in a batch, i.e.
		# "batch_mode": "truncate_episodes".
		if worker.config.batch_mode == "truncate_episodes":
			# Make sure this episode is really done.
			if hasattr(episode, "batch_builder"):
				assert episode.batch_builder.policy_collectors["default_policy"].batches[
					-1
				]["dones"][-1], (
					"ERROR: `on_episode_end()` should only be called "
					"after episode is done!"
				)
		# # add averages to custom metrics
		# response_time = np.mean(episode.user_data["response_times"])
		# episode.custom_metrics["response_times_avg"] = response_time
		# add to hist data
		for key in self.RELEVANT_KEYS:
			if key in episode.user_data:
				episode.hist_data[key] = episode.user_data[key]
				episode.custom_metrics[key] = episode.user_data[key]

		# add worker index
		episode.hist_data["worker_index"] = episode.user_data["worker_index"]

	def on_sample_end(
		self, 
		*, 
		worker: RolloutWorker, 
		samples: SampleBatch, 
		**kwargs
	):
		# TODO: any sanity check to perform?
		pass

	def on_train_result(self, *, algorithm, result: dict, **kwargs):
		#generate a random number, then use it to extract one element of each array from result['custom_metrics'], then save them to file
		first_key = list(result['custom_metrics'].keys())[0]
		random_index = np.random.randint(0, len(result['custom_metrics'][first_key]))
		random_custom_metrics = {}
		for key in self.RELEVANT_KEYS:
			if key in result['custom_metrics']:
				random_custom_metrics[key] = result['custom_metrics'][key][random_index]

		simulation_folder = algorithm.config['logger_config']['logdir']

		it_id = self.iteration_id

		os.makedirs(f"{simulation_folder}/custom_metrics", exist_ok = True)
		filename = f"{simulation_folder}/custom_metrics/iteration_{it_id}.json"

		with open(filename, "w+") as f:
			f.write(json.dumps(random_custom_metrics, indent=2))
		
		print("Iteration {} ended".format(self.iteration_id))
		self.iteration_id += 1

		# you can mutate the result dict to add new fields to return
		result['callback_ok'] = True

	def on_learn_on_batch(
		self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
	) -> None:
		result['sum_actions_in_train_batch'] = train_batch['actions'].sum()
		# # Log the sum of actions in the train batch.
		# print(
		#     "policy.learn_on_batch() result: {} -> sum actions: {}".format(
		#         policy, result['sum_actions_in_train_batch']
		#     )
		# )

	def on_postprocess_trajectory(
		self,
		*,
		worker: RolloutWorker,
		episode: Episode,
		agent_id: str,
		policy_id: str,
		policies: Dict[str, Policy],
		postprocessed_batch: SampleBatch,
		original_batches: Dict[str, Tuple[Policy, SampleBatch]],
		**kwargs,
	):
		pass