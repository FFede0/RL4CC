from algorithms.generators.ppo_config_generator import PPOConfigGenerator
from experiments.train import TrainingExperiment
from utilities.common import load_config_file, compare_dictionaries
from utilities.logger import Logger
import os

def test_default_generator(
    logger: Logger, expected_out_dir: str
  ):
  # run
  ppogenerator = PPOConfigGenerator(logger=logger)
  dd = ppogenerator.to_dict(ppogenerator.base_algo_config)
  # load expected output for comparison
  expected_dd = load_config_file(
    os.path.join(expected_out_dir, "test_default_generator.json")
  )
  equal = compare_dictionaries(dd, expected_dd)
  if not equal:
    raise RuntimeError("train_PPO.test_default_generator() test failed")

def test_generator(
    logger: Logger, exp_config_file: str, expected_out_dir: str
  ):
  # load config files
  exp_config = load_config_file(exp_config_file)
  env_config = load_config_file(exp_config["env_config_file"])
  ray_config = load_config_file(exp_config["ray_config_file"])
  # run
  ppogenerator = PPOConfigGenerator(logger=logger)
  algo_config = ppogenerator.generate_algo_config(
    env_config=env_config,
    ray_config=ray_config,
    base_logdir=os.path.abspath(exp_config.get("logdir", "~/ray_results")),
    eval_interval=exp_config.get("evaluation_interval")
  )
  dd = ppogenerator.to_dict(algo_config)
  # load expected output for comparison
  expected_dd = load_config_file(
    os.path.join(expected_out_dir, "test_generator.json")
  )
  equal = compare_dictionaries(dd, expected_dd)
  if not equal:
    raise RuntimeError("train_PPO.test_generator() test failed")

def test_training_loop(
    logger: Logger, exp_config_file: str, expected_out_dir: str
  ):
  # run
  exp = TrainingExperiment(logger=logger, exp_config_file=exp_config_file)
  exp.run()
  # compare with expected output
  

def main():
  logger = Logger(name="RL4CC-RegressionTests-trainPPO")
  exp_config_file = "config_files/regression_tests/train_PPO/exp_config.json"
  expected_out_dir = "utilities/regression_tests/train_PPO_expected_outputs"
  #
  logger.breakline()
  logger.log("PPOConfigGenerator.base_algo_config")
  logger.breakline()
  test_default_generator(logger, expected_out_dir)
  #
  logger.breakline()
  logger.log("PPOConfigGenerator.generate_algo_config()")
  logger.breakline()
  test_generator(logger, exp_config_file, expected_out_dir)
  #
  logger.breakline()
  logger.log("TrainingExperiment.run()")
  logger.breakline()
  test_training_loop(logger, exp_config_file, expected_out_dir)
