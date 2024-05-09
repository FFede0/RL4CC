from utilities import regression_tests


def main():
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


if __name__ == "__main__":
  main()
