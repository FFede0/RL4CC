# RL4CC

The **R**einforcement **L**earning for the **C**omputing **C**ontinuum module 
provides a common interface to define environments and RL algorithms based 
on the [Ray RLLib](https://docs.ray.io/en/latest/rllib/index.html) library.

It includes the following components:

- a simple [`Environment`](environment/base_environment.py) implementation, 
that should be used as base class when defining more complex problems. It 
is created by loading the parameters included in the 
[`env_config.json` file](config_files/README.md#environment-configuration-file).

- an [`Algorithm`](algorithms/algorithm.py) class, to be used as entrypoint to 
define training experiments, supported by a factory of Ray `AlgorithmConfig` 
[generators](algorithms/generators_factory.py).

- a `Tuner` class, to be used as entrypoint to define automatic hyperparameter 
tuning (TBA)

- a `Logger`, that can be used to print `INFO`, `WARNING` and `ERROR` messages 
in a standard format.

Detailed information about these components are provided in the following 
sections.

## How to start a training experiment

To define and start a training experiment exploiting one of the available 
algorithms:
1. in a suitable directory, define the `env_config.json`, `ray_config.json` 
and `exp_config.json` files as detailed [here](config_files/README.md)
2. build an `Algorithm` object by providing the path to the directory of 
configuration files
3. call the `Algorithm.training_loop()`

Example (for the PPO algorithm, using the provided `BaseEnvironment` and 
assuming configuration files to be saved in `./config_files/test1`):

```
PPO = Algorithm("PPO")
PPO.build(BaseEnvironment, "config_files/test1")
PPO.training_loop()
```

### Expected outputs

The outputs produced during the training experiment are saved in a suitable sub-directory of the `logdir` specified in the 
[`exp_config.json` file](config_files/README.md#experiment-configuration-file) 
(or of `~/ray_results` if nothing is provided). These include:

- `complete_config`: a directory reporting the three configuration files 
used to define the experiment. Note that, while `env_config.json` and 
`exp_config.json` are simply copied from the user-specified input directory, 
the `ray_config.json` file reported here includes also the default values 
assigned to keys that were not included in the user-defined configuration file.

- `exp_progress.json`: a file that, during the training, is progressively 
updated with information related to the last executed iteration, the last 
saved checkpoint, etc. It also reports the start and end timestamps of the 
experiment, its duration (in seconds) and the average length (in seconds) of 
each training iteration.

- `checkpoints`: a directory with checkpoints saved according to the frequency 
specified in the 
[`exp_config.json` file](config_files/README.md#experiment-configuration-file). 
Regardless the specified interval, a checkpoint is always saved at the end of 
the training process.

- `evaluations.txt`: each row of this file is a dictionary with values 
collected during the evaluation phase, which runs according 
to the frequency specified in the 
[`exp_config.json` file](config_files/README.md#experiment-configuration-file). 
The dictionary structure follows the one described for the progress.csv file, 
with an additional field specifying after how many training iterations it has 
been run.

- `figures`: a directory with plots generated during the training, according 
to the frequency specified in the 
[`exp_config.json` file](config_files/README.md#experiment-configuration-file).

- `progress.csv` and/or `result.json`, according to the logging configuration 
specified in the 
[`ray_config.json` file](config_files/README.md#ray-algorithm-configuration-file). 
Each row of these files includes values collected during one 
training iteration. By default, this will store:
  - values related to the environment and agent behaviour, as, e.g., the 
  minimum, maximum and average observed reward, the episode length, the number 
  of observed episodes, etc.
  - values related to the Ray cluster status and the resources usage, as, 
  e.g., the number of healthy workers, the percentage of CPU utilization, the 
  execution time, etc.
  - custom values specified by properly implementing the training callbacks 
  (see, e.g., the provided 
  [`BaseCallbacks` class](callbacks/base_callbacks.py)).

## How to start hyperparameter tuning

TBA

## The RL4CC Logger

The RL4CC `Logger` can be configured to print messages with different 
verbosity levels, using either the `sys.stdout`/`err` streams or 
suitably-defined file streams according to the information specified in the 
[`exp_config.json` file](config_files/README.md#configure-experiment-logging).

The format of logged messages is:

```
{TIME} [{LOGGER_NAME}] (level {LEVEL}) {MESSAGE_TYPE}: {MESSAGE}
```

where:
- `TIME` is given by `datetime.datetime.now()`.
- The `LOGGER_NAME` is provided as parameter in the `Logger` constructor 
(default: `RL4CCLogger`).
- The message `LEVEL` is 0 for warnings and errors (which are always printed 
regardless the verbosity level specified by the user), while it is specified 
as parameter when calling the `Logger.log()` method for generic messages. As 
mentioned in the 
[`exp_config.json` file](config_files/README.md#configure-experiment-logging), 
generic messages are printed only if the corresponding `LEVEL` is lower than 
the verbosity imposed by the user.
- The `MESSAGE_TYPE` is `INFO` when calling `Logger.log()`, `WARNING` when 
calling `Logger.warn()` and `ERROR` when calling `Logger.error()`.

## How to add new RL methods

To expand the module with generators for new algorithms:
1. implement a suitable subclass of the base 
[`AlgoConfigGenerator`](algorithms/generators/algo_config_generator.py) 
(see, as an example, what is provided for the 
[PPO algorithm](algorithms/generators/ppo_config_generator.py))
2. add the new generator to the 
[generators factory](algorithms/generators_factory.py)
