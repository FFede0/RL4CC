"""
Copyright 2026 Federica Filippini

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
from RL4CC.utilities.common import load_config_file, NumpyEncoder
from RL4CC.log_and_report.rl4cc_logger import Logger
from RL4CC.algorithms.algorithm import Algorithm

from ray.rllib.connectors.agent.obs_preproc import ObsPreprocessorConnector
from ray.rllib.utils.typing import AgentConnectorDataType
from ray.rllib.policy.sample_batch import (
  DEFAULT_POLICY_ID, 
  MultiAgentBatch, 
  SampleBatch
)

from gymnasium.spaces.utils import flatten as flatten_gym_observation
from gymnasium.spaces import Dict as gdict, Tuple as gtuple
from typing import Dict, Any, List, Optional
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from threading import RLock
import numpy as np
import traceback
import datetime
import json
import os

APP_BOOTSTRAP_MODULES = os.environ.get("APP_BOOTSTRAP_MODULES", "")
if APP_BOOTSTRAP_MODULES:
  for module in APP_BOOTSTRAP_MODULES.split(","):
    module = module.strip()
    if module:
      __import__(module)

APP_BOOTSTRAP_MODULES = os.environ.get("APP_BOOTSTRAP_MODULES", "")
if APP_BOOTSTRAP_MODULES:
  for module in APP_BOOTSTRAP_MODULES.split(","):
    module = module.strip()
    if module:
      __import__(module)


##############################################################################
# CONFIG
##############################################################################
class Settings:
  def __init__(self) -> None:
    self.LOG_FILE_BASE = os.getenv("LOG_FILE_BASE")  # optional
    self.INPUT_CHECKPOINT_DIR = os.environ.get("INPUT_CHECKPOINT_DIR")
    self.OUTPUT_CHECKPOINT_DIR = os.environ.get("OUTPUT_CHECKPOINT_DIR")
    self.CONFIGURATION_FILE = os.environ.get("CONFIGURATION_FILE")
    # check required fields
    required = {
      "INPUT_CHECKPOINT_DIR": self.INPUT_CHECKPOINT_DIR,
      "CONFIGURATION_FILE": self.CONFIGURATION_FILE
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
      raise RuntimeError(
        f"Missing required environment variables: {', '.join(missing)}"
      )
    # open configuration
    self.config = load_config_file(self.CONFIGURATION_FILE)

settings = Settings()


##############################################################################
# HELPER FUNCTIONS
##############################################################################
def build_dqn_replay_buffer(body, policy) -> MultiAgentBatch:
  """
  Build the legacy RLlib DQN replay batch from a generic JSON request.

  ``SampleBatch`` has fields specific to the algorithm and Ray version. Keeping
  their construction here makes the HTTP endpoint independent of DQN internals
  and leaves a clear extension point for PPO, SAC, or custom algorithms.
  """
  # check dimension
  num_transitions = len(body.observations)
  if num_transitions == 0:
    raise ValueError("at least one transition is required")
  if len(body.next_observations) != num_transitions:
    raise ValueError(
      "observations and next_observations must have the same number of items"
    )
  if len(body.actions) != num_transitions:
    raise ValueError(
      f"expected {num_transitions} actions, received {len(body.actions)}"
    )
  if len(body.rewards) != num_transitions:
    raise ValueError(
      f"expected {num_transitions} rewards, received {len(body.rewards)}"
    )
  # flatten observation spaces to obtain a DQN batch
  try:
    observations = np.stack([
      flatten_training_observation(observation, policy)
      for observation in body.observations
    ])
    next_observations = np.stack([
      flatten_training_observation(observation, policy)
      for observation in body.next_observations
    ])
  except ValueError:
    raise
  except Exception as exc:
    raise ValueError(
      "observations cannot be stacked into a DQN batch"
    ) from exc
  # DQN in RLlib requires a discrete action space.  Do not silently turn an
  # invalid request into a different action by clipping or offsetting it.
  action_space = policy.action_space
  if not hasattr(action_space, "n"):
    raise ValueError("the DQN train endpoint requires a discrete action space")
  actions = np.asarray(body.actions, dtype=np.int64).reshape(-1)
  if np.any(actions < 0) or np.any(actions >= action_space.n):
    raise ValueError(
      f"DQN actions must be integer policy ids in [0, {action_space.n})"
    )
  terminateds = request_vector(
    body.terminateds, num_transitions, "terminated flags", np.bool_, False
  )
  truncateds = request_vector(
    body.truncateds, num_transitions, "truncated flags", np.bool_, False
  )
  timesteps = request_vector(
    body.timesteps, num_transitions, "timesteps", np.int64, 0
  )
  if body.timesteps is None:
    timesteps = np.arange(num_transitions, dtype=np.int64)
  episode_ids = request_vector(
    body.episode_ids, num_transitions, "episode ids", np.int64, body.episode_id
  )
  infos = body.infos if body.infos is not None else [
    {} for _ in range(num_transitions)
  ]
  if len(infos) != num_transitions:
    raise ValueError(
      f"expected {num_transitions} infos, received {len(infos)}"
    )
  # define batch
  sample_batch = SampleBatch({
    SampleBatch.OBS: observations,
    SampleBatch.NEXT_OBS: next_observations,
    SampleBatch.ACTIONS: actions,
    SampleBatch.REWARDS: np.asarray(body.rewards, dtype=np.float32),
    SampleBatch.TERMINATEDS: terminateds,
    SampleBatch.TRUNCATEDS: truncateds,
    SampleBatch.INFOS: np.asarray(infos, dtype=object),
    SampleBatch.EPS_ID: episode_ids,
    SampleBatch.UNROLL_ID: np.arange(num_transitions, dtype=np.int64),
    SampleBatch.AGENT_INDEX: np.zeros(num_transitions, dtype=np.int64),
    SampleBatch.T: timesteps,
    "weights": np.ones(num_transitions, dtype=np.float32),
  })
  return MultiAgentBatch(
    policy_batches = {body.policy_id: sample_batch}, 
    env_steps = sample_batch.count
  )


def build_logger(tag: str) -> Logger:
  if settings.LOG_FILE_BASE:
    log_file = f"{settings.LOG_FILE_BASE}_{tag}_{date_to_string()}.log"
    check_create_dirs(log_file, True)
    log_stream = open(log_file, "a")
    logger = Logger(
      name = "RLAgentServer", out_stream = log_stream, verbose = 2
    )
    return logger
  return Logger(name = "RLAgentServer", verbose = 4)


def check_create_dirs(file: str, create: bool = True) -> None:
  """
  Function to check if a file lies in an existing directory
  and, if desired, create the missing directories.

  Parameters
  ----------
  file: str
    The name of the file to be checked.

  create: bool (default = True)
    If True, the missing directories are created.
    If False, an error is raised.

  Raises
  ------
  NotADirectoryError
    The input file lies in a non existing directory.
  """
  # name of the directory
  directory = os.path.dirname(file)
  # check if it does not exist
  if (directory != "") and (not os.path.isdir(directory)):
    # create the directories
    if create: os.makedirs(directory)
    # else raise an error
    else:
      raise NotADirectoryError(
        f"The directory {directory}, where the file {file} should"
        " be created. does not exist."
      )


def coerce_json_observation(observation_space, value: Any) -> Any:
  """
  Restore JSON lists to tuples where a Gymnasium Tuple space requires it
  """
  if isinstance(observation_space, gdict) and isinstance(value, dict):
    if set(value) != set(observation_space.spaces):
      # Preserve an invalid mapping so ``contains`` can reject missing and
      # unexpected keys instead of silently discarding client-provided data.
      return value
    return {
      key: coerce_json_observation(space, value[key])
      for key, space in observation_space.spaces.items()
      if key in value
    }
  if isinstance(observation_space, gtuple) and isinstance(value, list):
    return tuple(
      coerce_json_observation(space, item)
      for space, item in zip(observation_space.spaces, value)
    )
  return value


def close_logger(logger: Logger) -> None:
  if settings.LOG_FILE_BASE:
    logger.out_stream.close()


def date_to_string() -> str:
  """
  Function to convert the current date and time into a string
  with the format %Y-%m-%d_%H-%M-%S.%f
  """
  # get the current date
  x = datetime.datetime.now()
  # convert the datetime object to string of specific format
  datetimeStr = x.strftime("%Y-%m-%d_%H-%M-%S.%f")
  return datetimeStr


def flatten_training_observation(raw_observation: Any, policy) -> np.ndarray:
  """
  Validate and flatten one JSON observation for an RLlib policy.

  The serving API deliberately does not define an observation schema.  Instead,
  the policy checkpoint is the source of truth: its Gymnasium observation space
  accepts arbitrary nested Dict/Tuple/Box/etc. structures and determines the
  flattened representation expected by the policy.
  """
  observation_space = getattr(
    policy.observation_space, "original_space", policy.observation_space
  )
  observation = coerce_json_observation(observation_space, raw_observation)
  if not observation_space.contains(observation):
    raise ValueError(
      "observation does not match the policy observation space "
      f"{observation_space!r}"
    )
  try:
    return np.asarray(flatten_gym_observation(observation_space, observation))
  except Exception as exc:
    raise ValueError(
      "could not flatten observation using the policy observation space"
    ) from exc


def format_observation(unformatted_obs: dict) -> dict:
  # format observation as required by the algorithm
  observation = {}
  for agent in algo.algo_config.policies:
    agent_policy = algo.get_policy(agent)
    try:
      obs_space = algo.algo.env_runner.spaces[agent][0] # [1] is action space, 
                                                        # [0] is obs space.
    except AttributeError:
      obs_space = agent_policy.observation_space
    # -- multi-agent setting
    if agent != DEFAULT_POLICY_ID:
      # ---- dict-like spaces
      if unformatted_obs.get(agent) is None:
          # Skip agents with missing observation.
          continue

      if isinstance(obs_space, dict) or isinstance(obs_space, gdict):
        for obs_key, obs_val in unformatted_obs[agent].items():
          if agent not in observation:
            observation[agent] = {}
          observation[agent][obs_key] = np.array(obs_val).reshape(
            obs_space[obs_key].shape
          )

        # The obs dict must be converted to 1D array.
        observation[agent] = preprocess(observation[agent], agent_policy)
      # ----
      elif len(unformatted_obs[agent]) == 1:
        obs_val = list(unformatted_obs[agent].values())[0]
        observation[agent] = np.array(obs_val).reshape(obs_space.shape)

    else:
      # ---- dict-like spaces
      if isinstance(obs_space, dict) or isinstance(obs_space, Dict):
        for obs_key, obs_val in unformatted_obs.items():
          observation[obs_key] = np.array(obs_val).reshape(
            obs_space[obs_key].shape
          )
      # ----
      else:
        if len(unformatted_obs) == 1:
          obs_val = list(unformatted_obs.values())[0]
          observation = np.array(obs_val).reshape(obs_space.shape)
  return observation


def preprocess(obs, policy):
  """
  Return the given observation preprocessed using the policy's preprocessing 
  pipeline.

  The main use case of this function is to convert a dict observation space to 
  a flatten 1D array. The output can be passed directly to the model.
  """
  pp = policy.agent_connectors[ObsPreprocessorConnector]
  pp = pp[0]
  _input_dict = {"obs": obs}
  acd = AgentConnectorDataType("0", "0", _input_dict)
  pp.reset(env_id="0")
  ac_o = pp([acd])[0]
  obs_pp = ac_o.data["obs"]
  return obs_pp


def request_vector(
    values: Optional[List[Any]], n: int, name: str, dtype, default
  ) -> np.ndarray:
  """
  Return one request field as a vector of exactly ``n`` values
  """
  if values is None:
    return np.full(n, default, dtype=dtype)
  vector = np.asarray(values, dtype=dtype).reshape(-1)
  if vector.size != n:
    raise ValueError(f"expected {n} {name}, received {vector.size}")
  return vector


##############################################################################
# SCHEMA(s)
##############################################################################
class ActionRequest(BaseModel):
  observation: Dict[str, Any]
  agent_parameters: Dict[str, Any] = {}  # optional, defaults to empty dict


class TrainRequest(BaseModel):
  """
  Offline transitions used for one or more online DQN policy updates.

  Observations intentionally use ``Any``: their JSON structure is validated
  against the restored policy's observation space at request time, rather than
  against a serving-app-specific model.
  """
  observations: List[Any]
  actions: List[Any]
  rewards: List[float]
  next_observations: List[Any]
  terminateds: Optional[List[bool]] = None
  truncateds: Optional[List[bool]] = None
  infos: Optional[List[Dict[str, Any]]] = None
  episode_id: int = 0
  episode_ids: Optional[List[int]] = None
  timesteps: Optional[List[int]] = None
  policy_id: str = DEFAULT_POLICY_ID
  gradient_steps: int = 1
  update_target: bool = True


class CheckpointRequest(BaseModel):
  """
  Configure how the current in-memory agent should be checkpointed.
  """
  manual: bool = False


##############################################################################
# APP & ALGORITHM
##############################################################################
app = FastAPI()
algo = Algorithm(
  settings.config["algorithm"], checkpoint_path = settings.INPUT_CHECKPOINT_DIR
)
last_iteration = algo.last_iteration()
algo_lock = RLock()

REPLAY_BUFFER_BUILDERS = {
  "DQN": build_dqn_replay_buffer,
}

##############################################################################
# ROUTES
##############################################################################
@app.get("/")
def is_alive():
  return {
    "message": "The agent is up & running!",
    "endpoints": ["/action", "/train", "/checkpoint"],
  }


@app.post("/action")
def post_action_request(body: ActionRequest):
  """
  Function getting a json with the current environment state and returning
  the next agent(s) action(s).
  """
  # create a logger
  logger = build_logger("action")
  try:
    # retrieve the content of the request
    logger.log("Reading request...", 0)
    unformatted_obs = body.observation
    agent_parameters = body.agent_parameters
    logger.log(
      f"  Request content --> {unformatted_obs}; {agent_parameters}", 2
    )
    logger.log("Request read.", 1)
    # extract the state and the agent parameters
    logger.log("Formatting observation...", 0)
    observation = format_observation(unformatted_obs)
    logger.log("Observation formatted.", 1)
    logger.log(f"  Formatted observation: {observation}", 2)
    # require action
    explore = agent_parameters.get("explore", False)
    logger.log(f"Interrogating the agent (explore={explore})...", 0)
    # A training request mutates the same policy object, so action computation
    # and training must not access it concurrently.
    with algo_lock:
      action = algo.compute_single_action(observation, explore)
    logger.log(f"Chosen action: {action}", 2)
    logger.log("Agent interrogated.", 1)
    dec_action = JSONResponse(
      content = json.loads(json.dumps(action, cls = NumpyEncoder))
    )
    return dec_action
  except Exception as e:
      logger.err(f"Exception: {e}")
      logger.err(traceback.format_exc())
      raise HTTPException(status_code = 500, detail = str(e))
  finally:
    close_logger(logger)


@app.post("/checkpoint")
def post_checkpoint_request(body: CheckpointRequest):
  """
  Persist the current in-memory agent and return its server-side path.
  """
  logger = build_logger("checkpoint")
  try:
    checkpoint_path = os.path.join(
      settings.OUTPUT_CHECKPOINT_DIR, str(last_iteration)
    )
    os.makedirs(checkpoint_path, exist_ok = True)
    with algo_lock:
      checkpoint_path = algo.save_checkpoint(
        manual = body.manual, path = checkpoint_path
      )
    return {
      "status": "Checkpoint saved.",
      "checkpoint_path": checkpoint_path,
      "manual": body.manual,
    }
  except Exception as exc:
    logger.err(f"Exception: {exc}")
    logger.err(traceback.format_exc())
    raise HTTPException(status_code = 500, detail = str(exc))
  finally:
    close_logger(logger)


@app.post("/train")
def post_train_request(body: TrainRequest):
  """
  Update the loaded policy with client-supplied transition data.

  The endpoint is deliberately strict about the declared algorithm.  A replay
  batch created for one algorithm must never be passed to a different one.
  """
  logger = build_logger("train")
  try:
    algo_to_train = settings.config["algorithm"]
    replay_builder = REPLAY_BUFFER_BUILDERS.get(algo_to_train)
    if replay_builder is None:
      raise ValueError(
        f"online training is not implemented for algorithm {algo_to_train}"
      )
    if body.gradient_steps < 1:
      raise ValueError("gradient_steps must be at least one")
    logger.log(
      f"Receiving {len(body.observations)} transition(s) for {algo_to_train}"
    )
    # train
    with algo_lock:
      policy = algo.get_policy(body.policy_id)
      if policy is None:
        raise ValueError(f"unknown policy_id: {body.policy_id}")
      replay_batch = replay_builder(body, policy)
      dqn_batch = replay_batch.policy_batches[body.policy_id]
      training_results = [
        policy.learn_on_batch(dqn_batch)
        for _ in range(body.gradient_steps)
      ]
      if body.update_target and hasattr(policy, "update_target"):
        policy.update_target()
      last_iteration += 1
    # prepare response
    response = {
      "status": "Training completed.",
      "algorithm": algo_to_train,
      "last_iteration": last_iteration,
      "policy_id": body.policy_id,
      "transitions": dqn_batch.count,
      "gradient_steps": body.gradient_steps,
      "training_results": training_results,
    }
    return JSONResponse(
      content=json.loads(json.dumps(response, cls=NumpyEncoder))
    )
  except ValueError as exc:
    logger.err(f"Invalid training request: {exc}")
    raise HTTPException(status_code = 422, detail = str(exc))
  except Exception as exc:
    logger.err(f"Exception: {exc}")
    logger.err(traceback.format_exc())
    raise HTTPException(status_code = 500, detail = str(exc))
  finally:
    close_logger(logger)
