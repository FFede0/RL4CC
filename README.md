# RL4CC

The **R**einforcement **L**earning for the **C**omputing **C**ontinuum module 
provides a common interface to define environments and RL algorithms based 
on the [Ray RLLib](https://docs.ray.io/en/latest/rllib/index.html) library.

It includes the following components:

- a simple [`Environment`](environment/base_environment.py) implementation, that
  should be used as a base class when defining more complex problems. It is
  created by loading the parameters included in the [`env_config`
  configuration](config_files/README.md#environment-configuration).

- an [`Algorithm`](algorithms/algorithm.py) class, used to define RL 
algorithms for training/hyperparameter tuning experiments, supported by a 
factory of Ray `AlgorithmConfig` 
[generators](algorithms/generators_factory.py).

- a simple [`Callbacks`](callbacks/base_callbacks.py) implementation, 
that should be used as base class when defining more complex problems.

- two simple [custom neural network models](models), based on PyTorch and 
TensorFlow, that can be used as starting points to implement more complex 
networks if needed.

- a [`TrainingExperiment`](experiments/train.py) class, to be used as 
entrypoint to define training experiments, as explained in 
the following [section](#how-to-start-a-training-experiment).

- an [`Tuner`](algorithms/tuner.py) class, used to define hyperparameter 
tuning experiments, supported by the `Algorithm` class that works as 
trainable and by a `TuneConfig` and `RunConfig` 
[generator](algorithms/generators/tune_config_generator.py).

- a [`TuningExperiment`](experiments/tune.py) class, to be used as entrypoint 
to define automatic hyperparameter tuning, as explained in 
the following [section](#how-to-start-hyperparameter-tuning).

- a `Logger`, that can be used to print `INFO`, `WARNING` and `ERROR` messages 
in a standard format.

Detailed information about these components are provided in the following 
sections.

## How to start a training experiment

To define and start a training experiment exploiting one of the available algorithms:

1. define the `exp_config` configuration (and, if no previous checkpoint is
   provided, the `env_config` and `ray_config` configurations) as detailed [in
   the README](config_files/README.md). These configurations can be defined in
   Python as dictionaries or using JSON files.

2. initialize a `TrainingExperiment` object by passing the `env_config`
   configuration or a path to the `exp_config.json` file.

3. call the `TrainingExperiment.run()` method.

Example using the predefined  `BaseEnvironment` and `BaseCallbacks` classes and
with a JSON file for `exp_config`:

```
from RL4CC.experiments.train import TrainingExperiment

<<<<<<< HEAD
exp = TrainingExperiment("config_files/exp_config.json")
=======
exp = TrainingExperiment(exp_config_file="config_files/exp_config.json")
>>>>>>> origin/test
exp.run()
```

### Training experiments with a custom Environment

To use a custom Environment implementation, this needs to be registered in the 
`ray.tune.registry`. As an example, suppose that your code directory follows 
the structure:

```
.
├── RL4CC
├── src
│   ├── __init__.py
│   └── my_custom_environment.py
└── main.py
```

and that the `src/__init__.py` file includes, similarly to the one reported 
here for the base Environment,

```
from .my_custom_environment import MyCustomEnvironment
from ray.tune.registry import register_env

register_env("MyCustomEnvironment", lambda config: MyCustomEnvironment(config))
```

To guarantee that the environment is properly loaded when starting the 
experiment, your `main.py` file should include:

```
import src
from RL4CC.experiments.train import TrainingExperiment

exp = TrainingExperiment(exp_config_file="config_files/exp_config.json")
exp.run()
```

i.e., you must ensure that `src/__init__.py` is actually executed. 

### Training experiments with a custom Model

To use a custom neural network, this needs to be registered in the 
`ray.rllib.models.ModelCatalog`. As an example, suppose that your code 
directory follows the structure:

```
.
├── RL4CC
├── src
│   ├── __init__.py
│   └── my_custom_model.py
└── main.py
```

and that the `src/__init__.py` file includes, similarly to the one reported 
in the `models` directory here,

```
from .my_custom_model import MyCustomModel
from ray.rllib.models import ModelCatalog

ModelCatalog.register_custom_model("my_custom_model", MyCustomModel)
```

To guarantee that the model is properly loaded when starting the 
experiment, your `main.py` file should include:

```
import src
from RL4CC.experiments.train import TrainingExperiment

<<<<<<< HEAD
exp = TrainingExperiment("config_files/exp_config.json")
=======
exp = TrainingExperiment(exp_config_file="config_files/exp_config.json")
>>>>>>> origin/test
exp.run()
```

i.e., you must ensure that `src/__init__.py` is actually executed. 

<<<<<<< HEAD
Moreover, the `custom_model` section of the ray_config.json file must be 
properly defined, as detailed in the corresponding 
=======
Moreover, the `custom_model` section of the `ray_config` configuration must be
properly defined, as detailed in the corresponding
>>>>>>> origin/test
[README](config_files/README.md#how-to-use-custom-policy-models).

### Expected outputs

The outputs produced during the training experiment are saved in a suitable
sub-directory of the `logdir` specified in the [`exp_config`
config](config_files/README.md#experiment-configuration) (or in `~/ray_results`
if nothing is provided). These include:

- `complete_config`: a directory containing the configuration (`exp_config`,
  `env_config` and `ray_config`) used to define the experiment, saved as JSON
  files.
> [!NOTE]
> Two important notes:
>   - JSON files are saved here, regardless the fact that the user passed the
>     configuration(s) as file(s) or as `dict` object(s).
>   - While `env_config.json` and `exp_config.json` are simply copied from the
>     user-defined configurations, the `ray_config.json` file reported here
>     includes also the default values assigned to keys that were not included
>     in the user-defined configuration.

- `exp_progress.json`: a file that, during the training, is progressively 
updated with information related to the last executed iteration, the last 
saved checkpoint, etc. It also reports the start and end timestamps of the 
experiment, its duration (in seconds) and the average length (in seconds) of 
each training iteration.

- `checkpoints`: a directory with checkpoints saved according to the frequency
  specified in the [`exp_config`
  config](config_files/README.md#experiment-configuration). Regardless the
  specified interval, a checkpoint is always saved at the end of the training
  process.

- `evaluations.txt`: each row of this file is a dictionary with values collected
  during the evaluation phase, which runs according to the frequency specified
  in the [`exp_config` config](config_files/README.md#experiment-configuration).
  The dictionary structure follows the one described for the progress.csv file,
  with an additional field specifying after how many training iterations it has
  been run.

- `figures`: a directory with plots generated during the training, according to
  the frequency specified in the [`exp_config`
  config](config_files/README.md#experiment-configuration).

- `progress.csv` and/or `result.json`, according to the logging configuration
  specified in the [`ray_config`
  config](config_files/README.md#ray-algorithm-configuration). Each row of these
  files includes values collected during one training iteration. By default,
  this will store:

  - values related to the environment and agent behaviour, as, e.g., the
    minimum, maximum and average observed reward, the episode length, the number
    of observed episodes, etc.

  - values related to the Ray cluster status and the resources usage, as, e.g.,
    the number of healthy workers, the percentage of CPU utilization, the
    execution time, etc.

  - custom values specified by properly implementing the training callbacks
    (see, e.g., the provided [`BaseCallbacks`
    class](callbacks/base_callbacks.py)).

## How to start hyperparameter tuning

Hyperparameter Tuning is an integration of the Ray Tune, Air, Rllib libraries.

<<<<<<< HEAD
To define and start a tuning experiment exploiting one of the available 
algorithms:
1. define the `tune_config.json` file in the `exp_config.json` file as 
indicated [in the README](config_files/README.md); note that, since the 
tuning experiment will run multiple training experiments, also the 
`env_config.json` and `ray_config.json` files need to be defined as 
described in the [previous section](#how-to-start-a-training-experiment).
2. initialize a `TuningExperiment` object by providing the path to the 
`exp_config.json` file;
3. call the `TrainingExperiment.run()` method.
  - Note that, since Air's RunConfig is used on top of the algorithm object, 
  the user can provide a list of callbacks (classes) as parameters to the run 
  method, overwriting any previous callbacks indicated.

Example (when using the pre-defined `BaseEnvironment` and, possibly, custom 
callbacks classes):
=======
To define and start a tuning experiment exploiting one of the available
algorithms:

1. define the `tune_config` configuration in the `exp_config` configuration as
   indicated [in the README](config_files/README.md); note that, since the
   tuning experiment will run multiple training experiments, also the
   `env_config` and `ray_config` configurations need to be defined as described
   in the [previous section](#how-to-start-a-training-experiment). You can
   define `tune_config` as a dictionary or create a JSON file like
   `tune_config.json`.

2. initialize a `TuningExperiment` object by providing the `exp_config`
   configuration, as a dictionary or as a path to an `exp_config.json` file.

3. call the `TrainingExperiment.run()` method.
  > [!NOTE]
  > Note that, since Air's RunConfig is used on top of the algorithm object, the
  > user can provide a list of callbacks (classes) as parameters to the run
  > method, overwriting any previous callbacks indicated.

Example when using the predefined `BaseEnvironment`, a `exp_config` given as
file and, possibly, custom callbacks classes:
>>>>>>> origin/test

```
from RL4CC.experiments.tune import TuningExperiment

# Basic usage:
<<<<<<< HEAD
exp = TuningExperiment("config_files/exp_config.json")
=======
exp = TuningExperiment(exp_config_file="config_files/exp_config.json")
>>>>>>> origin/test
exp.run()

# Optional callbacks:
from Mycallbacks import Mycallbacks1, Mycallbacks2 #(It is recommended to extend the BaseCallbacks class of RL4CC)
<<<<<<< HEAD
exp = TuningExperiment("config_files/exp_config.json")
=======
exp = TuningExperiment(exp_config_file="config_files/exp_config.json")
>>>>>>> origin/test
callbacks = [Mycallbacks, Mycallbacks2]
exp.run(callbacks=callbacks)
```

## The RL4CC Logger

The RL4CC `Logger` can be configured to print messages with different verbosity
levels, using either the `sys.stdout`/`err` streams or suitably-defined file
streams according to the information specified in the [`exp_config`
config](config_files/README.md#configure-experiment-logging).

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
  mentioned in the [`exp_config`
  config](config_files/README.md#configure-experiment-logging), generic messages
  are printed only if the corresponding `LEVEL` is lower than the verbosity
  imposed by the user.
- The `MESSAGE_TYPE` is `INFO` when calling `Logger.log()`, `WARNING` when 
calling `Logger.warn()` and `ERROR` when calling `Logger.error()`.

> [!WARNING]
> file streams TBA

## How to add new RL methods

To expand the module with generators for new algorithms:
1. implement a suitable subclass of the base 
[`AlgoConfigGenerator`](algorithms/generators/algo_config_generator.py) 
(see, as an example, what is provided for the 
[PPO algorithm](algorithms/generators/ppo_config_generator.py))
2. add the new generator to the 
[generators factory](algorithms/generators_factory.py)

## How to contribute

### Regression tests

WIP (do not touch yet)
