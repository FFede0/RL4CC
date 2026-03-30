# RL4CC

The **R**einforcement **L**earning for the **C**omputing **C**ontinuum module
provides a common interface to define environments and RL algorithms based
on the [Ray RLLib](https://docs.ray.io/en/releases-2.20.0/rllib/index.html) library[^1].

It includes the following components:

- two simple environments, that should be used as base classes when defining 
  more complex problems. They are created by loading the parameters included in 
  the [`env_config` configuration](RL4CC/config_files/README.md#environment-configuration).
  The two represent a [single-agent](RL4CC/environment/base_environment.py) 
  and a [multi-agent](RL4CC/environment/base_multiagent_environment.py) 
  environment, respectively.

- an [`Algorithm`](RL4CC/algorithms/algorithm.py) class, used to define RL
  algorithms for training/hyperparameter tuning experiments, supported by a
  factory of Ray `AlgorithmConfig`
  [generators](RL4CC/algorithms/generators_factory.py).

- a simple [`Callbacks`](RL4CC/callbacks/base_callbacks.py) implementation,
  that should be used as base class when defining more complex problems.

- two simple [custom neural network models](RL4CC/models), based on PyTorch and
  TensorFlow, that can be used as starting points to implement more complex
  networks if needed.

- a [`TrainingExperiment`](RL4CC/experiments/train.py) class, to be used as
  entrypoint to define training experiments, as explained in
  the following [section](#how-to-start-a-training-experiment).

- an [`Tuner`](RL4CC/algorithms/tuner.py) class, used to define hyperparameter
  tuning experiments, supported by the `Algorithm` class that works as
  trainable and by a `TuneConfig` and `RunConfig`
  [generator](RL4CC/algorithms/generators/tune_config_generator.py).

- a [`TuningExperiment`](RL4CC/experiments/tune.py) class, to be used as 
  entrypoint to define automatic hyperparameter tuning, as explained in
  the following [section](#how-to-start-hyperparameter-tuning).

- a simple [`ProgressReporter`](RL4CC/log_and_report/base_tune_progress_reporter.py) 
  for Ray Tune, which periodically logs information related to the number of 
  executed trials, the hardware resources usage and the optimization process 
  on the `exp_progress.json` file (see the section on 
  [expected outputs](#expected-outputs)), instead of writing them on the 
  console. To use the provided reporter (or a similar user-defined one), 
  configure it through the `tune_config` dictionary or JSON file as explained 
  in the [README](RL4CC/config_files/README.md#tune-configuration).

- a [`Logger`](RL4CC/log_and_report/rl4cc_logger.py), that can be used to 
  print `INFO`, `WARNING` and `ERROR` messages in a standard format.

Detailed information about these components are provided in the following
sections.

Additionally, RL4CC enables to serve pre-trained RL models (available as 
checkpoints) through a Web API that can be deployed as a Docker container. 
Additional information are provided in the 
["serve"](#use-rl4cc-to-serve-pre-trained-rl-agents) section.

## Build the RL4CC library

To build the RL4CC library, the RL4CC module needs to be installed as a package,
so that its classes and functions can imported with `from RL4CC.x.y import z`.
To install RL4CC as a package, place yourself in the repo main directory (at the 
same level as the `setup.py`), with the virtual environment that contains the 
RL4CC dependencies activated.

Then use:  
```
pip3 install .
```
to install RL4CC as a package.  
Now by checking the installed packages with `pip3 freeze`, you will notice that 
RL4CC is among the dependencies. 

> [!NOTE]
> To download and install the RL4CC library using `pip`, add to your 
> requirements file `git+https://github.com/FFede0/RL4CC.git` (or 
> `git+https://github.com/FFede0/RL4CC.git@test_ray2.40.0` to install a 
> specific branch version).

## How to start a training experiment

To define and start a training experiment exploiting one of the available 
algorithms:

1. define the `exp_config` configuration (and, if no previous checkpoint is
   provided, the `env_config` and `ray_config` configurations) as detailed [in
   the README](RL4CC/config_files/README.md). These configurations can be 
   defined in Python as dictionaries or using JSON files.

2. initialize a `TrainingExperiment` object by passing the `env_config`
   configuration or a path to the `exp_config.json` file.

3. call the `TrainingExperiment.run()` method.

Example using the predefined `BaseEnvironment` and `BaseCallbacks` classes and
with a JSON file for `exp_config`:

```
from RL4CC.experiments.train import TrainingExperiment

exp = TrainingExperiment(exp_config_file="config_files/exp_config.json")
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

exp = TrainingExperiment(exp_config_file="config_files/exp_config.json")
exp.run()
```

i.e., you must ensure that `src/__init__.py` is actually executed.

Moreover, the `custom_model` section of the `ray_config` configuration must be
properly defined, as detailed in the corresponding
[README](RL4CC/config_files/README.md#how-to-use-custom-policy-models).

### Training experiments with plots

If you want to automatically generate plots during the training, you can use the 
`TrainingExperimentWithPlots` class, which is a subclass of `TrainingExperiment`. 
This class will automatically generate plots the last iteration and a moving 
average of all the iterations. The plots will be saved in the `plots` directory 
in the `logdir` previously specified.
In order to specify the plots to be generated, you can use the `RELEVANT_KEYS` of 
the callbacks: in particular, define a custom callback class (extending the 
`BaseCallbacksForPlots`class).
As we use `custom_metrics` to save all metrics that we want to plot, you should set

```
"reporting": {
  "keep_per_episode_custom_metrics": true
}
```

in the `ray_config` configuration.

### Expected outputs

The outputs produced during the training experiment are saved in a suitable
sub-directory of the `logdir` specified in the [`exp_config`
config](RL4CC/config_files/README.md#experiment-configuration) (or in `~/ray_results`
if nothing is provided). These include:

- `complete_config`: a directory containing the configuration (`exp_config`,
  `env_config` and `ray_config`) used to define the experiment, saved as JSON
  files.

> [!NOTE]
> Two important notes:
>
> - JSON files are saved here, regardless the fact that the user passed the
>   configuration(s) as file(s) or as `dict` object(s).
> - While `env_config.json` and `exp_config.json` are simply copied from the
>   user-defined configurations, the `ray_config.json` file reported here
>   includes also the default values assigned to keys that were not included
>   in the user-defined configuration.

- `exp_progress.json`: a file that, during the training, is progressively
  updated with information related to the last executed iteration, the last
  saved checkpoint, etc. It also reports the start and end timestamps of the
  experiment, its duration (in seconds) and the average length (in seconds) of
  each training iteration.

- `checkpoints`: a directory with checkpoints saved according to the frequency
  specified in the [`exp_config`
  config](RL4CC/config_files/README.md#experiment-configuration). Regardless the
  specified interval, a checkpoint is always saved at the end of the training
  process.

- `evaluations.json`: a json file containing key "evaluations", which is an
  array of dictionaries with values collected
  during the evaluation phase, which runs according to the frequency specified
  in the [`exp_config` config](RL4CC/config_files/README.md#experiment-configuration).
  The dictionary structure follows the one described for the progress.csv file,
  with an additional field specifying after how many training iterations it has
  been run.

- `figures`: a directory with plots generated during the training, according to
  the frequency specified in the [`exp_config`
  config](RL4CC/config_files/README.md#experiment-configuration).

- `progress.csv` and/or `result.json`, according to the logging configuration
  specified in the [`ray_config`
  config](RL4CC/config_files/README.md#ray-algorithm-configuration). Each row of these
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
    class](RL4CC/callbacks/base_callbacks.py)).

## How to start hyperparameter tuning

Hyperparameter Tuning is an integration of the Ray Tune, Air, Rllib libraries.

To define and start a tuning experiment exploiting one of the available
algorithms:

1. define the `tune_config` configuration in the `exp_config` configuration as
   indicated [in the README](RL4CC/config_files/README.md); note that, since the
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

Example when using the predefined `BaseEnvironment` and a `exp_config` given as
file:

```
from RL4CC.experiments.tune import TuningExperiment

# Basic usage:
exp = TuningExperiment(exp_config_file="config_files/exp_config.json")
exp.run()
```

## The RL4CC Logger

The RL4CC `Logger` can be configured to print messages with different verbosity
levels, using either the `sys.stdout`/`err` streams or suitably-defined file
streams according to the information specified in the [`exp_config`
config](RL4CC/config_files/README.md#configure-experiment-logging).

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
  config](RL4CC/config_files/README.md#configure-experiment-logging), generic messages
  are printed only if the corresponding `LEVEL` is lower than the verbosity
  imposed by the user.
- The `MESSAGE_TYPE` is `INFO` when calling `Logger.log()`, `WARNING` when
  calling `Logger.warn()` and `ERROR` when calling `Logger.error()`.

> [!WARNING]
> file streams TBA

## How to add new RL methods

To expand the module with generators for new algorithms:
1. implement a suitable subclass of the base 
[`AlgoConfigGenerator`](RL4CC/algorithms/generators/algo_config_generator.py) 
(see, as an example, what is provided for the 
[PPO algorithm](RL4CC/algorithms/generators/ppo_config_generator.py))
2. add the new generator to the 
[generators factory](RL4CC/algorithms/generators_factory.py)

The RL4CC library implements the MAPPO algorithm, i.e., a PPO version that 
exploits a centralized critic model. Details about the MAPPO algorithm 
configuration parameters are provided in the 
[algorithms/README](algorithms/README.md) file.

## Use RL4CC to serve pre-trained RL agents

RL4CC provides a ready-to-use REST API (based on 
[FastAPI](https://fastapi.tiangolo.com)) to serve trained agents.

### Deployment instructions

A pre-configured [Dockerfile](deployment/Dockerfile.rl4cc-serve) is available 
to deploy the service without needing to install RL4CC manually. The 
required steps are:

1. enter the directory from which you want to run the application (more 
information are provided below)
2. download the Dockerfile:

```
curl -L \
  https://raw.githubusercontent.com/FFede0/RL4CC/test_ray2.40.0/deployment/Dockerfile.rl4cc-serve \
  -o Dockerfile
```

3. build the image:

```
docker build -t rl4cc/serve:26.03.27 .
```

4. run the container:

```
CHECKPOINT_DIR=path_to_the_trained_model
CONFIGURATION_FILE=my_exp_file.json

docker run \
  -d \
  --name agentserver \
  --rm \
  -p 8000:8000 \
  -v ${CHECKPOINT_DIR}:/app/models/checkpoint \
  -v ${CONFIGURATION_FILE}:/app/config/config.json \
  --shm-size=1g \
  rl4cc/serve:26.03.27
```

The service requires the following environment variables:
- `CHECKPOINT_DIR`: path to the trained model checkpoint; by default, it is 
set to `/app/models/checkpoint`, but it can be overwritten by adding 
`-e CHECKPOINT_DIR=some_other_path` to the `docker run` command (**note:** 
in this case, the destination folder when mounting the volume should be 
updated accordingly)
- `CONFIGURATION_FILE`: experiment configuration file (ideally in the same 
format discussed in the 
[README](RL4CC/config_files/README.md#experiment-configuration)); by default, 
it is set to `/app/config/config.json`, but it can be overwritten by adding 
`-e CONFIGURATION_FILE=some_other_path` to the `docker run` command (**note:** 
in this case, the destination folder when mounting the volume should be 
updated accordingly).

Once running, the service is available at: `http://${HOST}:${PORT}`. The 
`HOST` and `PORT` environment variables are set at container creation to 
`0.0.0.0` and `8000`, respectively, but can be overwritten by providing 
`-e HOST=<some valid ip address> -e PORT=<some available port>` to the 
`docker run` command.

> [!NOTE] 
> The only information currently loaded from the `CONFIGURATION_FILE` is the 
> algorithm name; therefore, a simplified version with respect to the full 
> experiment configuration file may be provided.

> [!CAUTION]
> In order to properly load the checkpoint, the service may require to install 
> specific libraries, import agent-specific modules, and/or register a custom 
> model or environment. 
> The user must therefore: (1) add to the repository from which the Docker 
> image is created a `requirements.txt` file with additional libraries to be 
> installed via `pip` (if any). (2) Ensure that all modules to be loaded, as 
> well as additional files required by the agent (e.g., datasets) are 
> available to the container; to this end, consider that any Python file in 
> the base directory from which you create the Docker image is copied in the 
> container working directory `/app`. To ensure that any custom environment or 
> model is properly registered, follow the instructions below.

If the pre-trained algorithm makes use of a custom environment, the module 
including instructions for the environment registration as those listed 
[above](#training-experiments-with-a-custom-environment):

```
from .my_custom_environment import MyCustomEnvironment
from ray.tune.registry import register_env

register_env("MyCustomEnvironment", lambda config: MyCustomEnvironment(config))
```

must be imported **before** the checkpoint is loaded (the same happens for a 
[custom neural network model](#training-experiments-with-a-custom-model)). To 
ensure this, set the environment variable `APP_BOOTSTRAP_MODULES` to the list 
of modules including registration instructions when creating the container. As 
an example, if the lines reported above are saved in a `register_env.py` file, 
add `-e APP_BOOTSTRAP_MODULES=register_env` to your `docker run` command.

> [!NOTE]
> About using Ray from Docker containers: as mentioned in 
> [the official documentation](https://docs.ray.io/en/latest/ray-overview/installation.html#launch-ray-in-docker), 
> Docker containers running Ray require access to reasonable memory. Replace 
> `--shm-size=1g` from the `docker run` command above with a limit appropriate 
> for your system. As mentioned, "A good estimate for this is to use roughly 
> 30% of your available memory (this is what Ray uses internally for its 
> Object Store)".

### Execution instructions

To check that the service has been correctly deployed, you can run:

```
HOST=0.0.0.0
PORT=8000

curl -X GET http://${HOST}:${PORT}/
```

This lists the available endpoints. In particular,

#### Compute a single action

The `/action` endpoint allows to ask the agent(s) the next action to take, 
given the current observation. 

It expects a body including an `observation` dictionary with the current 
observation the agent(s) should consider, and an `agent_parameters` including 
additional configuration information. The currently available configuration 
parameter is `explore`, to be set to `True` if the agent(s) is free to select 
a random action (with some probability that depends on the loaded 
algorithm), `False` if the agent(s) should fully exploit the loaded policy.

The `observation` format must match the observation space used during 
training. It supports both single and multi-agent setups (examples are 
provided below), and the provided inputs are automatically converted to 
NumPy arrays and reshaped according to the `observation_space` definition 
available in the environment.

Example (single-agent), following the 
[base environment](RL4CC/environment/base_environment.py):

```
curl -X POST http://localhost:8000/action \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {
      "current_time": 100
    },
    "agent_parameters": {
      "explore": false
    }
  }'
```

Example (multi-agent), following the 
[base environment](RL4CC/environment/base_multiagent_environment.py):

```
curl -X POST http://localhost:8000/action \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {
      "agent_0": {
        "current_time": 100
      },
      "agent_1": {
        "current_time": 100
      }
    },
    "agent_parameters": {
      "explore": false
    }
  }'
```

The response is given as a dictionary with a key for each agent (in the 
multi-agent setting) and the corresponding chosen action.

## How to contribute to RL4CC

The RL4CC repository is organized as follows:
- the branch `main` hosts production-ready releases, i.e. tested code that
has passed reviews on lower stages;
- the branch `develop` hosts changes that may not be completely stable.
This is basically a quality/staging branch. When the changes have been tested
and are stable, we can make a PR to main;
- the branch `test` is the collector of the initial merges among all
developers. From here we move on to `develop`.

Each developer can create their own branch named `test-[your-initials]`, 
from which you can merge to `test`. No direct PR will be accepted on any 
branch that is not `test`.

### Regression tests

Regression tests for algorithm generators and training experiments with Ray 
versions 2.8.1, 2.10.0 and 2.20.0 are available among the 
[utilities](RL4CC/utilities/regression_tests/README.md). 

[^1] The RL4CC library has been developed and tested considering Ray RLLib 
versions up to 2.20.0. Carefully select an appropriate version of the 
official Ray documentation when looking for additional information.
