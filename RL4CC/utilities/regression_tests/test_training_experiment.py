from RL4CC.experiments.train import TrainingExperiment
from RL4CC.utilities.common import load_config_file, write_config_file
from RL4CC.utilities.common import compute_deviation
from RL4CC.utilities.logger import Logger

from typing import Tuple
import pandas as pd
import json
import os


def test_training_loop(
    logger: Logger, 
    exp_config_file: str, 
    expected_out: str, 
    criterion: str, 
    tol: float
  ) -> Tuple[bool, str]:
  # run
  exp = TrainingExperiment(logger=logger, exp_config_file=exp_config_file)
  exp.run()
  # compare with expected output
  progress = pd.read_csv(
    os.path.join(exp.logdir, "progress.csv"), on_bad_lines='skip'
  )
  expected_progress = pd.read_csv(expected_out)
  passed = False
  if len(progress) == len(expected_progress):
    columns = [
      "episode_reward_max",
      "episode_reward_min",
      "episode_reward_mean",
      "episode_len_mean",
      "episodes_this_iter",
      "num_agent_steps_sampled",
      "num_agent_steps_trained",
      "num_env_steps_sampled_this_iter",
      "num_env_steps_trained_this_iter"
    ]
    passed = True
    for col in columns:
      _, m, M, avg = compute_deviation(expected_progress[col], progress[col])
      if criterion == "min":
        if m > tol:
          logger.err(
            f"test_training_loop() failed with mininum deviation: {m}"
          )
          passed = False
      elif criterion == "max":
        if M > tol:
          logger.err(
            f"test_training_loop() failed with maxinum deviation: {M}"
          )
          passed = False
      elif criterion == "avg":
        if avg > tol:
          logger.err(
            f"test_training_loop() failed with average deviation: {avg}"
          )
          passed = False
      else:
        logger.err(f"undefined criterion: {criterion}")
        passed = False
  return passed, exp.logdir


def update_experiment_config(exp_config: dict, exp_logdir: str) -> bool:
  updated = False
  # load experiment progress
  exp_progress = load_config_file(
    os.path.join(exp_logdir, "exp_progress.json")
  )
  if exp_progress is not None:
    # add the checkpoint to the experiment configuration
    last_checkpoint_dir = exp_progress.get("last_checkpoint_dir")
    if last_checkpoint_dir is not None:
      _ = exp_config.pop("env_config_file")
      _ = exp_config.pop("ray_config_file")
      exp_config["from_checkpoint"] = last_checkpoint_dir
      updated = True
  return updated


def test_training_experiment(
    logger: Logger, 
    name: str, 
    base_config_folder: str, 
    base_output_folder: str,
    criterion: str,
    tol: float
  ) -> Tuple[int, int]:
  num_passed_tests = 0
  total_num_tests = 0
  # run training experiment
  exp_config_file = os.path.join(
    base_config_folder, f"{name}/exp_config.json"
  )
  exp_config = load_config_file(exp_config_file)
  logger.log(f"test experiment from {exp_config_file}")
  if exp_config is not None:
    expected_output_file = os.path.join(
      base_output_folder, f"{name}.csv"
    )
    if os.path.exists(expected_output_file):
      passed, exp_logdir = test_training_loop(
        logger = logger,
        exp_config_file = exp_config_file,
        expected_out = expected_output_file,
        criterion = criterion,
        tol = tol
      )
      num_passed_tests += int(passed)
    else:
      logger.err(
        f"expected output file {expected_output_file} not found"
      )
  else:
    logger.err(
      f"configuration file {exp_config_file} not found"
    )
  total_num_tests += 1
  # resume training experiment from last checkpoint
  if exp_config is not None:
    if update_experiment_config(exp_config, exp_logdir):
      exp_config_file = write_config_file(
        json.dumps(exp_config, indent=2), 
        "utilities/regression_tests/LOGDIR/",
        f"{name}_fromCheckpoint.json"
      )
      expected_output_file = os.path.join(
        base_output_folder, f"{name}_fromCheckpoint.csv"
      )
      if os.path.exists(expected_output_file):
        passed, _ = test_training_loop(
          logger = logger,
          exp_config_file = exp_config_file,
          expected_out = expected_output_file,
          criterion = criterion,
          tol = tol
        )
        num_passed_tests += int(passed)
      else:
        logger.err(
          f"expected output file {expected_output_file} not found"
        )
    else:
      logger.err(
        f"could not update the configuration from {exp_logdir}"
      )
  total_num_tests += 1
  return num_passed_tests, total_num_tests


def main(criterion: str = "avg", tol: float = 1e-5) -> Tuple[int, int]:
  logger = Logger(name="RL4CC-RegressionTests-TrainingExperiment")
  base_config_folder = "config_files/regression_tests/"
  base_output_folder = "utilities/regression_tests/expected_output/"
  # test the training experiment for each provided algorithm configuration
  num_passed_tests = 0
  total_num_tests = 0
  for name in os.listdir(base_config_folder):
    if name.endswith("training"):
      logger.breakline()
      passed, total = test_training_experiment(
        logger = logger,
        name = name,
        base_config_folder = base_config_folder,
        base_output_folder = base_output_folder,
        criterion = criterion,
        tol = tol
      )
      num_passed_tests += passed
      total_num_tests += total
  logger.breakline()
  if num_passed_tests < total_num_tests:
    logger.warn(f"ONLY {num_passed_tests}/{total_num_tests} TEST PASSED")
  else:
    logger.log(f"{num_passed_tests}/{total_num_tests} TEST PASSED")
  return num_passed_tests, total_num_tests
