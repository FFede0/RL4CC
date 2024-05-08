from algorithms.generators.dqn_config_generator import DQNConfigGenerator
from experiments.train import TrainingExperiment
from utilities.common import load_config_file, compare_dictionaries, write_config_file
from utilities.logger import Logger
import os

def test_default_generator(
    logger: Logger, expected_out_dir: str
  ):
  # run
  dqngenerator = DQNConfigGenerator(logger=logger)
  dd = dqngenerator.to_dict(dqngenerator.base_algo_config)
  # load expected output for comparison
  expected_dd = load_config_file(
    os.path.join(expected_out_dir, "test_default_generator.json")
  )
  equal = compare_dictionaries(dd, expected_dd)
  if not equal:
    raise RuntimeError("train_DQN.test_default_generator() test failed")

def test_generator(
    logger: Logger, exp_config_file: str, expected_out_dir: str
  ):
  # load config files
  exp_config = load_config_file(exp_config_file)
  env_config = load_config_file(exp_config["env_config_file"])
  ray_config = load_config_file(exp_config["ray_config_file"])
  # run
  dqngenerator = DQNConfigGenerator(logger=logger)
  algo_config = dqngenerator.generate_algo_config(
    env_config=env_config,
    ray_config=ray_config,
    base_logdir=os.path.abspath(exp_config.get("logdir", "~/ray_results")),
    eval_interval=exp_config.get("evaluation_interval")
  )
  dd = dqngenerator.to_dict(algo_config)
  # load expected output for comparison
  expected_dd = load_config_file(
    os.path.join(expected_out_dir, "test_generator.json")
  )
  equal = compare_dictionaries(dd, expected_dd)
  if not equal:
    raise RuntimeError("train_DQN.test_generator() test failed")

def test_training_loop(
    logger: Logger, exp_config_file: str, expected_out_dir: str
  ):
  # run
  exp = TrainingExperiment(logger=logger, exp_config_file=exp_config_file)
  exp.run()
  # compare with expected output

def main():
  logger = Logger(name="RL4CC-RegressionTests-trainDQN")
  exp_config_file = "config_files/regression_tests/train_DQN/exp_config.json"
  expected_out_dir = "utilities/regression_tests/train_DQN_expected_outputs"
  #
  logger.breakline()
  logger.log("DQNConfigGenerator.base_algo_config")
  logger.breakline()
  test_default_generator(logger, expected_out_dir)
  #
  logger.breakline()
  logger.log("DQNConfigGenerator.generate_algo_config()")
  logger.breakline()
  test_generator(logger, exp_config_file, expected_out_dir)
  #
  logger.breakline()
  logger.log("TrainingExperiment.run()")
  logger.breakline()
  test_training_loop(logger, exp_config_file, expected_out_dir)
