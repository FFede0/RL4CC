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
from utilities.common import load_config_file, write_config_file
from environment.environments_factory import EnvironmentsFactory
from callbacks.callbacks_factory import CallbacksFactory
from utilities.logger import Logger

from abc import ABC, abstractmethod
import numpy as np
import json
import os


class BaseExperiment(ABC):
  def __init__(
      self, 
      exp_config_file: str,
      environments_factory = EnvironmentsFactory,
      callbacks_factory = CallbacksFactory
    ):
    # load experiment configuration file
    self.exp_config = load_config_file(exp_config_file)
    if self.exp_config is None:
      raise RuntimeError(
        f"ERROR: file `{exp_config_file}` not found or invalid"
      )
    # initialize logger
    self.logger = Logger(name = "RL4CC")
    if "logger" in self.exp_config:
      verbosity = self.exp_config["logger"].get("verbosity", 0)
      self.logger.verbose = verbosity
    # base output directory
    self.logdir = os.path.abspath(
      self.exp_config.get("logdir", "~/ray_results")
    )
    # validate other parameters
    self.validate_experiment_configuration()
    # checkpoint/plot/evaluation intervals
    self.checkpoint_interval = self.exp_config.get(
      "checkpoint_interval", np.inf
    )
    self.plot_interval = self.exp_config.get(
      "plot_interval", np.inf
    )
    self.evaluation_interval = self.exp_config.get(
      "evaluation_interval", np.inf
    )
    # stopping criteria
    self.define_stopping_criteria()
    # save factories
    self.environments_factory = environments_factory
    self.callbacks_factory = callbacks_factory
  
  def validate_experiment_configuration(self):
    # if a previous checkpoint path is provided...
    if "from_checkpoint" in self.exp_config:
      self.checkpoint_path = self.exp_config["from_checkpoint"]
      self.env_config = None
      self.ray_config = None
      # ...environment and ray configurations (if provided) are ignored
      keys = ["env_config_file", "ray_config_file"]
      for key in keys:
        if key in self.exp_config:
          self.logger.warn(
            f"previous checkpoint provided; `{key}` is ignored"
          )
    # otherwise, the environment configuration file is mandatory
    else:
      if "env_config_file" not in self.exp_config:
        raise KeyError(
          "ERROR: provide `env_config_file` if no previous checkpoint is given"
        )
      self.checkpoint_path = None
      self.env_config = load_config_file(self.exp_config["env_config_file"])
      self.ray_config = load_config_file(
        self.exp_config.get("ray_config_file", "")
      )
  
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
  
  def plot_results(self, result: dict) -> str:
    pass
  
  @abstractmethod
  def define_stopping_criteria(self, exp_config: dict):
    pass
  
  @abstractmethod
  def run(self):
    pass

  @staticmethod
  def serialize_evaluation_metrics(evaluation_metrics: dict) -> dict:
    """
    Serialize the dictionary of evaluation metrics
    """
    em = {**evaluation_metrics}
    for key, val in evaluation_metrics["evaluation"]["hist_stats"].items():
      newval = []
      for x in val:
        if isinstance(x, np.ndarray):
          newval.append(x.tolist())
        else:
          newval.append(x)
      em["evaluation"]["hist_stats"][key] = newval
    return em
