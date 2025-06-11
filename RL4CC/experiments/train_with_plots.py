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

  def execute_before_training(self, algo: Algorithm):
    super().execute_before_training(algo)
    if self.logdir is None:
      self.logger.warn("logdir is not defined. Using default logdir.")
    else: 
      self.plots_folder = os.path.join(self.logdir, "plots")
      os.makedirs(self.plots_folder, exist_ok=True)
  
  def execute_after_training(self, algo: Algorithm):
    super().execute_after_training(algo)

    self.manage_evaluation_files()
    self.manage_custom_metrics_keys()
    if self.plot_interval == np.inf:
      self.plot_all_evaluations()
    else:
      self.logger.log("Plot interval was fixed - not plotting all the evaluations at the end of the experiment.", 1)
    
    #plot custom metrics
    custom_metrics_folder = os.path.join(self.logdir, "custom_metrics")
    if os.path.exists(custom_metrics_folder):
        custom_metrics_files = os.listdir(custom_metrics_folder)
        custom_metrics_files = [int(f.replace('iteration_', '').replace('.json', '')) for f in custom_metrics_files]
        custom_metrics_files.sort()
        if len(custom_metrics_files) > 0:
            # Loop through the keys of the first file, then create a plot for each key after putting together the data from all the files
            first_iteration = json.load(open(os.path.join(custom_metrics_folder, 'iteration_'+str(custom_metrics_files[0])+'.json')))
            all_iterations = {}
            os.makedirs(os.path.join(self.plots_folder, 'training'), exist_ok=True)
            for key in first_iteration.keys():
                all_iterations[key] = []
            for custom_metrics_file in custom_metrics_files:
                iteration = json.load(open(os.path.join(custom_metrics_folder, 'iteration_'+str(custom_metrics_file)+'.json')))
                for key in iteration.keys():
                    if key in all_iterations.keys():
                        if isinstance(iteration[key], list):
                            all_iterations[key] += iteration[key]
                        else:
                            all_iterations[key].append(iteration[key])

            for key in all_iterations.keys():
                data = all_iterations[key]
                plt.title(f'training: {key}')
                plt.plot(data)
                plt.savefig(os.path.join(self.plots_folder, 'training', f'{key}.png'), bbox_inches='tight', dpi=100)
                plt.close()
                plt.title(f'MOVING AVERAGE training: {key}')
                if (len(data) > 100):
                  window_size = len(data) // 100
                else:
                  window_size = 1
                window = np.ones(window_size) / window_size
                smoothed_data = np.convolve(data, window, mode='valid')
                plt.plot(smoothed_data)
                plt.savefig(os.path.join(self.plots_folder, 'training', f'{key}_moving_average.png'), bbox_inches='tight', dpi=100)
                plt.close()

        else:
            print('CUSTOM TRAIN: No custom metrics files found')
    else:
        print('CUSTOM TRAIN: No custom metrics folder found')


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
  
  def plot_results(self, result):
    super().plot_results(result)
    self.logger.log("Plotting evaluation results", 1)
    if not self.custom_metrics_keys:
      if 'evaluation' in result.keys() and 'custom_metrics' in result['evaluation'].keys():
        self.custom_metrics_keys = list(result["evaluation"]["custom_metrics"].keys())
    evaluation_folder = os.path.join(self.plots_folder, "evaluation"+str(result["training_iteration"]))
    os.makedirs(evaluation_folder, exist_ok=True)
    for key in self.custom_metrics_keys:
      evaluation_values = result["evaluation"]["custom_metrics"]
      if key in evaluation_values.keys():
        evaluation_values = evaluation_values[key]
        number_of_possible_configurations = len(evaluation_values)
        for configuration_id in range(number_of_possible_configurations):
          current_folder = evaluation_folder+'/'+str(configuration_id)
          os.makedirs(current_folder, exist_ok=True)
          if (isinstance(evaluation_values[configuration_id][0], list) or isinstance(evaluation_values[configuration_id][0], np.ndarray)) and not len(evaluation_values[configuration_id][0]) == 1:
            for i in range(len(evaluation_values[configuration_id][0])):
              values = [value[i] for value in evaluation_values[configuration_id]]
              plt.plot(values, label=f"{key}_{i}")
              plt.xlabel("time")
              plt.ylabel(key)
              plt.legend()
              plt.title(key)
              plt.savefig(
                os.path.join(current_folder, f"{key}_{i}.png")
              )
              plt.close()
          else:
            values = evaluation_values[configuration_id]
            plt.plot(values, label=f"{key}")
            plt.xlabel("time")
            plt.ylabel(key)
            plt.legend()
            plt.title(key)
            plt.savefig(
              os.path.join(current_folder, f"{key}.png")
            )
            plt.close()

  def plot_all_evaluations(self):
    if len(self.evaluations) == 0:
      self.logger.warn("No evaluations found.")
    else:
      for eval_idx in range(len(self.evaluations)):
        plots_eval_folder = os.path.join(self.plots_folder, "evaluation"+str(eval_idx+1))
        if not os.path.exists(plots_eval_folder):
          os.makedirs(plots_eval_folder)
        for key in self.custom_metrics_keys:
          last_evaluation_values = self.evaluations[eval_idx]["custom_metrics"][key]
          last_evaluation_multiple_values = False
          if isinstance(last_evaluation_values, list) and isinstance(last_evaluation_values[0], list):
            if (
                (isinstance(last_evaluation_values[0][0], list) and len(last_evaluation_values[0][0]) == 1) or
                  (isinstance(last_evaluation_values[0][0], np.ndarray) and len(last_evaluation_values[0][0])) == 1 or
                    isinstance(last_evaluation_values[0][0], int) or
                      isinstance(last_evaluation_values[0][0], float)
            ):
              last_evaluation_values = np.array(last_evaluation_values).flatten()
            elif (
                  (isinstance(last_evaluation_values[0][0], list) and len(last_evaluation_values[0][0]) > 1) or
                    (isinstance(last_evaluation_values[0][0], np.ndarray) and len(last_evaluation_values[0][0]) > 1)
            ):
              last_evaluation_multiple_values = True
            else:
              self.logger.err("Error: unknown type")
          else:
            self.logger.err(
              f"Error: custom metric {key} is not a list of lists"
            )

          plt.figure(key, figsize = (10, 10))
          if last_evaluation_multiple_values:
            # plot the different values in the same plot, so that if the array is [[1,2,3],[4,5,6],[7,8,9]] we have 3 lines with points (1,4,7), (2,5,8), (3,6,9)
            for i in range(len(last_evaluation_values[0][0])):
              values = [value[i] for value in last_evaluation_values[0]]
              # print('PLOTTING:', values)
              plt.plot(values, label=f"{key}_{i}")
          else:
            plt.plot(last_evaluation_values, label=key)

          plt.xlabel("time")
          plt.ylabel(key)
          plt.legend()
          plt.title(key)
          plt.savefig(
            os.path.join(plots_eval_folder, f"{key}_last_evaluation.png")
          )
          plt.close()
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
