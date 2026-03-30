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

from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from gymnasium.spaces import Dict as gdict
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import numpy as np
import datetime
import json
import os
import traceback

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
    self.CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR")
    self.CONFIGURATION_FILE = os.environ.get("CONFIGURATION_FILE")
    # check required fields
    required = {
      "CHECKPOINT_DIR": self.CHECKPOINT_DIR,
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


def format_observation(unformatted_obs: dict) -> dict:
  # format observation as required by the algorithm
  observation = {}
  for agent in algo.algo_config.policies:
    agent_policy = algo.get_policy(agent)
    obs_space = agent_policy.observation_space
    # -- multi-agent setting
    if agent != DEFAULT_POLICY_ID:
      # ---- dict-like spaces
      if isinstance(obs_space, dict) or isinstance(obs_space, gdict):
        for obs_key, obs_val in unformatted_obs[agent].items():
          if agent not in observation:
            observation[agent] = {}
          observation[agent][obs_key] = np.array(obs_val).reshape(
            obs_space[obs_key].shape
          )
      # ----
      else:
        if len(unformatted_obs[agent]) == 1:
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


##############################################################################
# SCHEMA(s)
##############################################################################
class ActionRequest(BaseModel):
  observation: Dict[str, Any]
  agent_parameters: Dict[str, Any] = {}  # optional, defaults to empty dict


##############################################################################
# APP & ALGORITHM
##############################################################################
app = FastAPI()
algo = Algorithm(
  settings.config["algorithm"], checkpoint_path = settings.CHECKPOINT_DIR
)

##############################################################################
# ROUTES
##############################################################################
@app.get("/")
def is_alive():
  return {
    "message": "The agent is up & running!",
    "endpoints": ["/action"],
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
    # require action
    logger.log("Interrogating the agent...", 0)
    action = algo.compute_single_action(
      observation, agent_parameters.get("explore", False)
    )
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


# @app.post("/train")
# def post_continue_training(body):
#   # create a logger
#   logger = build_logger("train")
#   try:
#     # retrieve the content of the design time file
#     logger.log("Receiving request...", 0)
#     experience_json = body.experience
#     logger.log(f"Request (i.e., experience) --> {experience_json}", 2)
#     logger.log("Request received.", 1)
#     # call the training procedure
#     train_agent(
#       experience = experience_json,
#       configuration_directory = os.path.dirname(settings.CONFIGURATION_FILE),
#       plot_directory = None,
#       logger = logger
#     )
#     return {"status": "Training completed."}
#   except Exception as e:
#     raise HTTPException(status_code=500, detail=str(e))
#   finally:
#     close_logger(logger)
