from RL4CC.utilities import regression_tests
from RL4CC.utilities.logger import Logger


def main():
  logger = Logger(name="RL4CC-RegressionTests")
  num_passed_tests = 0
  total_num_tests = 0
  # generators
  passed, total = regression_tests.test_generators.main()
  num_passed_tests += passed
  total_num_tests += total
  # training experiment
  passed, total = regression_tests.test_training_experiment.main()
  num_passed_tests += passed
  total_num_tests += total
  # print recap
  logger.breakline()
  if num_passed_tests < total_num_tests:
    logger.warn(f"ONLY {num_passed_tests}/{total_num_tests} TEST PASSED")
  else:
    logger.log(f"{num_passed_tests}/{total_num_tests} TEST PASSED")


if __name__ == "__main__":
  main()
