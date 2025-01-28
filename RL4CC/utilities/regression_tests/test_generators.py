from RL4CC.utilities.common import load_config_file, write_config_file
from RL4CC.utilities.common import compare_dictionaries
from RL4CC.utilities.logger import Logger
from RL4CC.algorithms.generators_factory import ACGfactory

from typing import Tuple
import os


def test_default_generator(
    logger: Logger, algo: str, expected_out: str
  ) -> bool:
  """
  Test the generation of a default `AlgorithmConfig` object for the given 
  method
  """
  # run
  generator = ACGfactory.create(algo, logger = logger)
  algo_config_dict = generator.to_dict(generator.base_algo_config)
  # load expected output for comparison
  expected_dict = load_config_file(expected_out)
  equal, different_keys = compare_dictionaries(algo_config_dict, expected_dict)
  if not equal:
    logger.err(
      f"failed test_default_generator() on algo: {algo}; different keys: {different_keys}"
    )
    write_config_file(
      generator.to_json(generator.base_algo_config),
      "utilities/regression_tests/ERRORS",
      f"{algo}_default_config.json"
    )
  return equal


def test_algo_config_generator(
    logger: Logger, exp_config_file: str, expected_out: str
  ) -> bool:
  """
  Test the generation of an `AlgorithmConfig` object based on the given 
  configuration files
  """
  # load config files
  exp_config = load_config_file(exp_config_file)
  env_config = load_config_file(exp_config["env_config_file"])
  ray_config = load_config_file(exp_config["ray_config_file"])
  algo = exp_config["algorithm"]
  # run
  generator = ACGfactory.create(algo, logger = logger)
  algo_config = generator.generate_algo_config(
    env_config = env_config,
    ray_config = ray_config,
    eval_interval = exp_config.get("evaluation_interval")
  )
  algo_config_dict = generator.to_dict(algo_config)
  # load expected output for comparison
  expected_dict = load_config_file(expected_out)
  equal, different_keys = compare_dictionaries(algo_config_dict, expected_dict)
  if not equal:
    logger.err(
      f"failed test_generator() on algo: {algo}; different keys: {different_keys}"
    )
    write_config_file(
      generator.to_json(algo_config),
      "utilities/regression_tests/ERRORS",
      f"{algo}_config.json"
    )
  return equal


def test_algo_generators(
    logger: Logger, algo: str, base_config_folder: str, base_output_folder: str
  ) -> Tuple[int, int]:
  num_passed_tests = 0
  total_num_tests = 0
  # default
  logger.log(f"test default generator for algo: {algo}")
  expected_output_file = os.path.join(
    base_output_folder, f"{algo}_default_config.json"
  )
  if os.path.exists(expected_output_file):
    passed = test_default_generator(
      logger = logger, 
      algo = algo, 
      expected_out = expected_output_file
    )
    num_passed_tests += int(passed)
  else:
    logger.err(
      f"the expected output file {expected_output_file} is not available"
    )
  total_num_tests += 1
  # generator based on configuration files
  exp_config_file = os.path.join(
    base_config_folder, f"{algo}generator/exp_config.json"
  )
  logger.log(f"test generator for algo: {algo} using {exp_config_file}")
  if os.path.exists(exp_config_file):
    expected_output_file = os.path.join(
      base_output_folder, f"{algo}_config.json"
    )
    if os.path.exists(expected_output_file):
      passed = test_algo_config_generator(
        logger = logger, 
        exp_config_file = exp_config_file, 
        expected_out = expected_output_file
      )
      num_passed_tests += int(passed)
    else:
      logger.err(
        f"the expected output file {expected_output_file} is not available"
      )
  else:
    logger.err(
      f"the configuration file {exp_config_file} is not available"
    )
  total_num_tests += 1
  return num_passed_tests, total_num_tests


def main() -> Tuple[int, int]:
  logger = Logger(name="RL4CC-RegressionTests-Generators")
  base_config_folder = "config_files/regression_tests/"
  base_output_folder = "utilities/regression_tests/expected_output/"
  # test the registered generators
  num_passed_tests = 0
  total_num_tests = 0
  for algo in ACGfactory.algo_config_generators.keys():
    logger.breakline()
    passed, total = test_algo_generators(
      logger = logger,
      algo = algo,
      base_config_folder = base_config_folder,
      base_output_folder = base_output_folder
    )
    num_passed_tests += passed
    total_num_tests += total
  logger.breakline()
  if num_passed_tests < total_num_tests:
    logger.warn(f"ONLY {num_passed_tests}/{total_num_tests} TEST PASSED")
  else:
    logger.log(f"{num_passed_tests}/{total_num_tests} TEST PASSED")
  return num_passed_tests, total_num_tests
