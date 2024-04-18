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
from algorithms.generators_factory import ACGfactory
from environment.base_environment import BaseEnvironment
from utilities.logger import Logger

from ray.rllib.algorithms.algorithm import Algorithm as RayAlgorithm
from datetime import datetime
from typing import Tuple
import numpy as np
import json
import os


class Algorithm:
  def __init__(self, algo: str):
    self.logger = Logger("RL4CC-Algorithm")
    self.algo_config_generator = ACGfactory.create(algo)
    self.algo = None
    self.checkpoint_interval = np.inf
    self.plot_interval = np.inf
    self.evaluation_interval = np.inf

  def build(
      self, environment: BaseEnvironment, config_dir: str
    ):
    """
    Build the `Algorithm` according to the provided environment class and 
    directory of configuration files
    """
    # load the configuration files from the provided directory
    env_config, ray_config, exp_config = self.load_config_files(config_dir)
    # load the Ray `Algorithm` from a checkpoint (if provided)
    if exp_config is not None and "from_checkpoint" in exp_config:
      chkpt_path = exp_config["from_checkpoint"]
      if not os.path.exists(chkpt_path) or not os.path.isdir(chkpt_path):
        raise FileNotFoundError(
          f"ERROR: checkpoint path {chkpt_path} does not exist or is invalid"
        )
      self.algo = RayAlgorithm.from_checkpoint(chkpt_path)
      logdir = self.algo.logdir
      self.logger.warn(
        f"Algorithm restored from checkpoint; the output directory is {logdir}"
      )
    # otherwise...
    else:
      # ...generate `AlgorithmConfig`
      algo_config = self.algo_config_generator.generate_algo_config(
        environment, env_config, ray_config, exp_config
      )
      # ...build and save the Ray `Algorithm`
      self.algo = algo_config.build()
    # define the stopping criterion
    self.define_stopping_criterion(exp_config)
    # save the configuration files
    self.write_config_files(env_config, exp_config, self.algo.logdir)
  
  def training_loop(self):
    """
    `Algorithm` training loop
    """
    start = datetime.now()
    self.logger.log(f"training loop --> START")
    self.update_progress_file("experiment_start_timestamp", start.timestamp())
    it = 1
    while not self.stop(it):
      # train
      self.logger.log(f"starting iteration {it}", 2)
      result = self.algo.train()
      self.logger.log("iteration completed", 2)
      self.update_progress_file("last_iteration", result['training_iteration'])
      # save checkpoint at the beginning and every `checkpoint_interval` 
      # iterations
      if it == 1 or it % self.checkpoint_interval == 0:
        self.save_checkpoint(result['training_iteration'])
      # plot results at the beginning and every `plot_interval` iterations
      if it == 1 or it % self.plot_interval == 0:
        self.plot_results(result)
      # save evaluation results every `evaluation_interval` iterations
      if it % self.evaluation_interval == 0:
        self.update_evaluation_metrics_file(
          result["training_iteration"], 
          result.evaluation_metrics
        )
      # move to the next iteration
      it += 1
    # save last checkpoint
    self.save_checkpoint(result['training_iteration'])
    # perform final evaluation
    self.logger.log(f"starting final evaluation", 1)
    self.update_evaluation_metrics_file(
      result["training_iteration"], self.algo.evaluate()
    )
    self.logger.log(f"final evaluation performed", 1)
    # stop
    self.algo.stop()
    end = datetime.now()
    self.update_progress_file("experiment_end_timestamp", end.timestamp())
    self.logger.log("training loop ---> END")
    # last progress update
    experiment_duration = end - start
    avg_time_per_iter = (end - start) / (it - 1)
    self.update_progress_file(
      "experiment_duration_s", experiment_duration.total_seconds()
    )
    self.update_progress_file(
      "avg_time_per_iter_s", avg_time_per_iter.total_seconds()
    )
    self.logger.log(f"training loop took: {experiment_duration}")
    self.logger.log(f"average time per iteration: {avg_time_per_iter}")
  
  def define_stopping_criterion(self, exp_config: dict):
    """
    Define a `stop()` function to check whether the training loop should be 
    terminated, according to the stopping criteria specified in the experiment 
    configuration file
    """
    # check that stopping criteria are provided
    if exp_config is None or "stopping_criteria" not in exp_config:
      raise KeyError(
        "`stopping_criteria` must be provided in `exp_config.json`"
      )
    # list possible stopping criteria
    stop_on_max_iter = None
    for key, value in exp_config["stopping_criteria"].items():
      if key == "max_iterations":
        stop_on_max_iter = lambda it : it > value
      else:
        raise NotImplementedError(
          f"Stopping criterion `{key}` is not supported"
        )
    self.stop = stop_on_max_iter
  
  def save_checkpoint(self, last_iteration: dict):
    """
    Save an `Algorithm` checkpoint (the checkpoint directory name is given 
    by the last iteration number)
    """
    save_result = self.algo.save(
      checkpoint_dir = os.path.join(
        self.algo.logdir, f"checkpoints/{last_iteration}"
      )
    )
    last_checkpoint_dir = save_result.checkpoint.path
    self.logger.log(
      "an Algorithm checkpoint has been created inside directory: "
      f"'{last_checkpoint_dir}'", 1
    )
    self.update_progress_file("last_checkpoint_dir", last_checkpoint_dir)
  
  def load_config_files(self, config_dir: str) -> Tuple[dict, dict, dict]:
    """
    Load the configuration files needed to build the `Algorithm`
    """
    # read environment configuration file (mandatory!)
    env_config = load_config_file(os.path.join(config_dir, "env_config.json"))
    if env_config is None:
      raise FileNotFoundError(f"env_config.json not found in {config_dir}")
    # read ray configuration file
    ray_config = load_config_file(os.path.join(config_dir, "ray_config.json"))
    # read experiment configuration file (and related information)
    exp_config = load_config_file(os.path.join(config_dir, "exp_config.json"))
    if exp_config is not None:
      self.checkpoint_interval = exp_config.get("checkpoint_interval", np.inf)
      self.plot_interval = exp_config.get("plot_interval", np.inf)
      self.evaluation_interval = exp_config.get("evaluation_interval", np.inf)
      # Logger verbosity
      if "logger" in exp_config:
        self.logger.verbose = exp_config["logger"].get("verbosity", 0)
    return env_config, ray_config, exp_config
  
  def print_algo_config(self, to_file: bool = True):
    """
    Print the `AlgorithmConfig` in json format (by default, to a file saved 
    in the `Algorithm` logdir)
    """
    jj = self.algo_config_generator.to_json(self.algo.config)
    if to_file:
      write_config_file(
        jj, 
        os.path.join(self.algo.logdir, "complete_config"), 
        "ray_config.json"
      )
    else:
      print(jj)
  
  def write_config_files(
      self, env_config: dict, exp_config: dict, logdir: str
    ):
    """
    Write the environment and experiment configuration files into the 
    algorithm logdir
    """
    # write environment configuration file
    write_config_file(
      json.dumps(env_config), 
      os.path.join(logdir, "complete_config"), 
      "env_config.json"
    )
    # write algorithm configuration file
    self.print_algo_config()
    # write experiment configuration file
    if env_config is not None:
      write_config_file(
        json.dumps(exp_config), 
        os.path.join(logdir, "complete_config"), 
        "exp_config.json"
      )
  
  def update_progress_file(self, key: str, value):
    """
    Update the information written in the experiment progress file
    """
    exp_progress = {}
    exp_progress_file = os.path.join(self.algo.logdir, "exp_progress.json")
    # load existing content (if any)
    if os.path.exists(exp_progress_file):
      with open(exp_progress_file, "r") as istream:
        exp_progress = json.load(istream)
    # update
    exp_progress[key] = value
    # write updated file
    with open(exp_progress_file, "w") as ostream:
      ostream.write(json.dumps(exp_progress, indent = 2))

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
    # write
    evaluation_file = os.path.join(self.algo.logdir, "evaluation.txt")
    with open(evaluation_file, "a") as ostream:
      ostream.write(f"{evaluation}\n")
  
  def serialize_evaluation_metrics(self, evaluation_metrics: dict) -> dict:
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

  def plot_results(self, result: dict) -> str:
    pass
