"""
Copyright 2024 Federica Filippini

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
from RL4CC.utilities.common import load_config_file, write_config_file
from RL4CC.utilities.common import not_defined, defined
from RL4CC.utilities.common import update_json_file
from RL4CC.utilities.common import NumpyEncoder
from RL4CC.utilities.logger import Logger

from ray.rllib.policy.policy import Policy
from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
import json
import os


class BaseExperiment(ABC):
  def __init__(self,
               exp_config_file: str = None,
               exp_config: dict = None,
               logger: Logger = Logger(name = "RL4CC")):
    # Handle exp_config_file and exp_config, they cannot be both None or both
    # set.
    if exp_config_file is None and exp_config is None:
      raise RuntimeError(
        "ERROR: exp_config_file and exp_config cannot both be None"
      )
    if exp_config_file is not None and exp_config is not None:
      raise RuntimeError(
        "ERROR: exp_config_file and exp_config cannot both be defined"
      )
    # Set self.exp_config.
    if exp_config_file is not None:
      self.exp_config = load_config_file(exp_config_file)
      if self.exp_config is None:
        raise RuntimeError(
          f"ERROR: file {exp_config_file!r} not found or invalid"
        )
    else:
      self.exp_config = exp_config
    # initialize logger
    self.logger = logger
    if "logger" in self.exp_config:
      verbosity = self.exp_config["logger"].get("verbosity", 0)
      self.logger.verbose = verbosity
    # validate other parameters
    self.validate_experiment_configuration()
    # checkpoint config
    self.define_checkpoint_config()
    # plot/evaluation intervals
    self.plot_interval = self.exp_config.get(
      "plot_interval", np.inf
    )
    self.evaluation_interval = self.exp_config.get(
      "evaluation_interval", np.inf
    )
    # stopping criteria
    self.define_stopping_criteria()

  def validate_experiment_configuration(self):
    # the algorithm name must be provided
    if not_defined("algorithm", self.exp_config):
      raise KeyError(
        "ERROR: `algorithm` is required"
      )
    # if a previous checkpoint path is provided...
    if "from_checkpoint" in self.exp_config:
      self.checkpoint_path = os.path.abspath(
        self.exp_config["from_checkpoint"]
      )
      self.env_config = None
      self.ray_config = None
      self.logdir = None
      # ...environment and ray configurations (if provided) are ignored
      keys = [
        "env_config_file",
        "env_config",
        "ray_config_file",
        "ray_config",
        "logdir"
      ]
      for key in keys:
        if key in self.exp_config:
          self.logger.warn(
            f"previous checkpoint provided; {key!r} is ignored"
          )
    # otherwise, the environment configuration file is mandatory
    else:
      # Load the env_config. The user can specify the env_config via the
      # env_config_file parameter or directly via the env_config parameter.
      if (not_defined("env_config_file", self.exp_config)
          and not_defined("env_config", self.exp_config)):
        raise KeyError(
          "ERROR: provide 'env_config_file' or 'env_config' if no previous "
          "checkpoint is given"
        )
      if (defined("env_config_file", self.exp_config)
          and defined("env_config", self.exp_config)):
        raise KeyError(
          "ERROR: 'env_config_file' or 'env_config' cannot be both set!"
        )
      if defined("env_config_file", self.exp_config):
        self.env_config = load_config_file(self.exp_config["env_config_file"])
      else:
        self.env_config = self.exp_config["env_config"]
      # Load the ray_config. The user can specify the ray_config via the
      # ray_config_file parameter or directly via the ray_config parameter.
      if (not_defined("ray_config_file", self.exp_config)
          and not_defined("ray_config", self.exp_config)):
        raise KeyError(
          "ERROR: provide 'ray_config_file' or 'ray_config' if no previous "
          "checkpoint is given"
        )
      if (defined("ray_config_file", self.exp_config)
          and defined("ray_config", self.exp_config)):
        raise KeyError(
          "ERROR: 'ray_config_file' or 'ray_config' cannot be both set!"
        )
      if defined("ray_config_file", self.exp_config):
        self.ray_config = load_config_file(self.exp_config["ray_config_file"])
      else:
        self.ray_config = self.exp_config["ray_config"]
      self.checkpoint_path = None
      # base output directory
      base_logdir = self.exp_config.get(
        "logdir"#, os.path.expanduser("~/ray_results"))
      )
      if base_logdir is not None:
        self.generate_logdir(
          base_logdir,
          self.exp_config.get("algorithm"),
          self.env_config.get("env_name")
        )
      else:
        self.logdir = None

  def generate_logdir(self, base_logdir: str, algo: str, env_name: str) -> str:
    """
    Generate the experiment `logdir` if an appropriate parameter is provided
    """
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    self.logdir = os.path.join(
      os.path.abspath(base_logdir), f"{algo}_{env_name}_{now}"
    )
    os.makedirs(self.logdir, exist_ok=True)

  def define_checkpoint_config(self):
    """
    Initialize the dictionary storing all configuration parameters related
    to checkpointing
    """
    self.checkpoint_config = {
      "checkpoint_frequency": self.exp_config.get(
        "checkpoint_interval", np.inf
      )
    }

  def write_config_files(self):
    """
    Write the environment and experiment configuration files into the
    experiment logdir
    """
    # write environment configuration file
    if self.env_config is not None:
      write_config_file(
        json.dumps(self.env_config, indent = 2),
        os.path.join(self.logdir, "complete_config"),
        "env_config.json"
      )
    # write experiment configuration file
    write_config_file(
      json.dumps(self.exp_config, indent = 2),
      os.path.join(self.logdir, "complete_config"),
      "exp_config.json"
    )

  def update_progress_file(self, key: str, value):
    """
    Update the information written in the experiment progress file
    """
    update_json_file(
      os.path.join(self.logdir, "exp_progress.json"), key, value
    )

  def update_evaluation_metrics_file(
      self, last_iter: int, evaluation_metrics: dict
    ):
    """
    Save the result of the last evaluation
    """
    # create the serialized dictionary of the last evaluation results
    evaluation = {
      "after_training_iteration": last_iter,
      **self.serialize_evaluation_metrics(evaluation_metrics)
    }
    # write the evaluation results in the evaluations file (json format)
    evaluations_dict = {'evaluations': []}

    evaluations_file_path = os.path.join(self.logdir, 'evaluations.json')
    if os.path.exists(evaluations_file_path):
      with open(evaluations_file_path, 'r') as f:
        evaluations_dict = json.load(f)

    evaluations_dict['evaluations'].append(evaluation)

    with open(evaluations_file_path, 'w+') as f:
      # We need to use a custom JSON encoder because the user can be added some
      # custom properties as NumPy arrays. NumPy types need to be converted to
      # Python types before doing the JSON dump because the default encoder
      # doesn't support NumPy types.
      json.dump(evaluations_dict, f, indent=4, cls=NumpyEncoder)

  def plot_results(self, result: dict) -> str:
    pass

  @abstractmethod
  def define_stopping_criteria(self, exp_config: dict):
    pass

  @abstractmethod
  def run(self) -> Policy:
    pass

  @staticmethod
  def serialize_evaluation_metrics(evaluation_metrics: dict) -> dict:
    """
    Serialize the dictionary of evaluation metrics
    """
    if "evaluation" in evaluation_metrics:
      evaluation_metrics = evaluation_metrics["evaluation"]
    em = {}
    items_for_loop = {}
    if (
      "env_runners" in evaluation_metrics.keys() and
        "hist_stats" in evaluation_metrics["env_runners"].keys()
    ):
      items_for_loop = evaluation_metrics["env_runners"]["hist_stats"].items()
      em = {**evaluation_metrics["env_runners"]}
    elif (
      "hist_stats" in evaluation_metrics.keys() and
        len(evaluation_metrics["hist_stats"].keys()) > 0
    ):
      items_for_loop = evaluation_metrics["hist_stats"].items()
      em = {**evaluation_metrics}
    for key, val in items_for_loop:
      newval = []
      for x in val:
        if isinstance(x, np.ndarray):
          newval.append(x.tolist())
        else:
          newval.append(x)
      em["hist_stats"][key] = newval
    return em
