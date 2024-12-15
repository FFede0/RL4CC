"""
Copyright 2024 Riccardo Cavadini, Federica Filippini

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
from RL4CC.experiments.train import TrainingExperiment
from RL4CC.algorithms.algorithm import Algorithm
from RL4CC.utilities.logger import Logger

import matplotlib.pyplot as plt
import numpy as np
import json
import os


class TrainingExperimentWithPlots(TrainingExperiment):
  def __init__(
      self,
      exp_config_file: str = None,
      exp_config: dict = None,
      logger: Logger = Logger(name = "RL4CC")
    ):
    super().__init__(exp_config_file, exp_config, logger)
    self.evaluations = []
    self.merged_evaluations = {}
    self.custom_metrics_keys = []
    if self.logdir is None:
      self.logger.warn("logdir is not defined. Using default logdir.")
    self.plots_folder = os.path.join(self.logdir, "plots")
  
  def execute_after_training(self, algo: Algorithm):
    if not os.path.exists(self.plots_folder):
      os.makedirs(self.plots_folder)
    self.manage_evaluation_files()
    self.manage_custom_metrics_keys()
    self.plot()
  
  def manage_evaluation_files(self):
    evaluations_dict = {"evaluations": []}
    # if evaluations are saved in a json file, load it to dictionary
    if os.path.exists(os.path.join(self.logdir, "evaluations.json")):
      with open(os.path.join(self.logdir, "evaluations.json")) as f:
        evaluations_dict = json.load(f)
        self.evaluations = evaluations_dict["evaluations"]
    # otherwise, perform the conversion from txt to json format
    else:
      with open(os.path.join(self.logdir, "evaluations.txt")) as f:
        for line in f.readlines():
          line = line.replace("\'", "\"")
          line = line.replace("True", "true")
          line = line.replace("False", "false")
          line = line.replace("None", "null")
          line_json = json.loads(line)
          self.evaluations.append(line_json)
      # remove the last evaluation (as an extra one is always saved) 
      # TODO: fix this at some point
      if (len(self.evaluations) > 1):
          self.evaluations.pop(-1)
      evaluations_dict["evaluations"] = self.evaluations
      with open(os.path.join(self.logdir, "evaluations.json"), "w") as f:
          json.dump(evaluations_dict, f, indent=4)
    # save the loaded information as a unique json
    if len(self.evaluations) == 0:
      self.logger.warn("No evaluations found.")
    else:
      self.custom_metrics_keys = list(
        self.evaluations[0]["custom_metrics"].keys()
      )
      for evaluation in self.evaluations:
        evaluation_custom_metrics = evaluation["custom_metrics"]
        for key in self.custom_metrics_keys:
          if key not in self.merged_evaluations.keys():
            self.merged_evaluations[key] = evaluation_custom_metrics[key]
          else:
            self.merged_evaluations[key].extend(evaluation_custom_metrics[key])
      with open(os.path.join(self.logdir,"merged_evaluations.json"), "w") as f:
        json.dump(self.merged_evaluations, f, indent=4)
  
  def manage_custom_metrics_keys(self):
      pass
  
  def plot(self):
    if len(self.evaluations) == 0:
      self.logger.warn("No evaluations found.")
    else:
      for key in self.custom_metrics_keys:
        last_evaluation_values = self.evaluations[-1]["custom_metrics"][key]
        if isinstance(last_evaluation_values, list):
          if (
              isinstance(last_evaluation_values[0], list) or 
                isinstance(last_evaluation_values[0], np.ndarray) or 
                  isinstance(last_evaluation_values[0], int) or 
                    isinstance(last_evaluation_values[0], float)
          ):
            last_evaluation_values = np.array(last_evaluation_values).flatten()
          else:
            self.logger.err("Error: unknown type")
        else:
          self.logger.err(
            f"Error: custom metric {key} is not a list of lists"
          )
        plt.figure(key, figsize = (10, 10))
        plt.plot(last_evaluation_values, label=key)
        plt.xlabel("time")
        plt.ylabel(key)
        plt.legend()
        plt.title(key)
        plt.savefig(
          os.path.join(self.plots_folder, f"{key}_last_evaluation.png")
        )
        plt.close()