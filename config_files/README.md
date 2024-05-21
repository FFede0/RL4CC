## Expected structure of the configuration files

Each experiment is controlled by one base configuration file, in `JSON` 
format, denoted in the following as `exp_config.json`. It includes information 
about the experiment to run (e.g., the name of the algorithm to use, 
whether to start from an existing checkpoint, and, possibly, details 
about the Ray `Tuner`). 

If the experiment should start from scratch (i.e., no previous checkpoints 
are available), the configuration is completed by two additional files, which 
provide information about the `Environment` and the Ray `Algorithm`.

The corresponding structure is detailed in the following.

### `Environment` configuration file

The `env_config.json` file includes **four** mandatory parameters, which 
are related to Environment name and the simulation time management within it. 

These are:
- `env_name`: the name of the Environment, as it is registered in the 
[`ray.tune.registry`](../environment/__init__.py);
- `min_time`: the start time of the simulation (in seconds);
- `max_time`: the end time of the simulation (in seconds);
- `time_step`: the time elapsed between two subsequent calls to 
`Environment.step()` (in seconds).

The interval `[min_time, max_time]` corresponds to an **episode**; therefore, 
a sample configuration as:

```
{
  "env_name": "BaseEnvironment",
  "min_time": 0,
  "max_time": 3600,
  "time_step": 10
}
```

can be used to represent a scenario where episodes last 1 hour each and a 
new agent decision is taken every 10 seconds.

**Important note:** these parameters should be set even in a non-episodic 
environment. Use `max_time` to represent the Environment horizon.

### Ray `Algorithm` configuration file

The `ray_config.json` file includes parameters related to the definition 
of a Ray [`AlgorithmConfig`](https://docs.ray.io/en/latest/rllib/rllib-training.html#configuring-rllib-algorithms) object. 

Parameters should be grouped in sub-dictionaries following the callbacks 
structure of `AlgorithmConfig`. The most relevant families of parameters are:
- `framework`, for the Deep Learning framework options;
- `rollouts`, for parameters related to the configuration of rollout workers, 
i.e., to how experience trajectories are collected;
- `evaluation`, to configure the `Algorithm` evaluation;
- `resources`, to determine which types and how many resources are devoted 
to experience collection and algorithm training;
- `training`, to set both common training parameters (e.g., the learning 
rate) and algorithm-specific properties.

A more comprehensive list is provided in 
[the Ray documentation](https://docs.ray.io/en/latest/rllib/rllib-training.html#configuring-rllib-algorithms).

:warning::warning::warning: **Important note:** to simplify the management of 
some parameters related to the experience sampling and training, RL4CC offers 
the possibility of setting higher-level *suggested* keywords insted of 
using directly the ones defined in Ray. These are:
- In the `rollouts` section:
  - `duration_unit`: it can take the value `episodes`, if rollout workers 
  should collect entire episodes during the experience sampling phase, or 
  `timesteps`, if episodes can be truncated during the experience sampling;
  - `duration_per_worker`: how many episodes/steps should be collected by 
  each rollout worker.
- In the `training` section:
  - `batch_size`: dimension of each batch extracted from the collected 
  experience (or the replay buffer, if defined) during the training phase;
  - `num_train_batches`: how many batches should be trained in each iteration.
- in the `resources` section:
  - `num_gpus_master`: number of GPUs assigned to the master node;
  - `num_cpus_master`: number of CPUs assigned to the master node.
- in the `evaluation` section:
  - `evaluation_duration_per_worker`: how many episodes/steps should be collected by each evaluation worker.

These keywords mask a lower-level management performed by Ray, where different 
algorithms use different parameters to control the same elements. An 
**expert** user is free to set directly the Ray *protected* keywords, but 
the two approaches cannot be mixed.

:warning::warning::warning: **Important note:** there are few elements that, 
differently from what is explained in the Ray documentation **should not** be 
managed through `ray_config.json`. These are:
- `env` and `env_config`, from the `environment` parameters group;
- `evaluation_interval`, from the `evaluation` parameters group;
- `logdir`, from the `logger_config` dictionary in the `debugging` parameters 
group.

In particular, `env` and `env_config` are indirectly controlled through the 
[`env_config.json`](#environment-configuration-file) file and the `Environment` 
passed to the algorithm (or the algorithm configuration) initializer, while `evaluation_interval` and `logdir` are set from the 
[experiment configuration file](#experiment-configuration-file).

**Additional note:** in the `callbacks` section, the `callbacks_class` 
parameter should correspond to the path to the callbacks class as it would 
be reported while importing the module (e.g., 
`"callbacks.base_callbacks.BaseCallbacks"`).

Sample `ray_config.json` files for [PPO](ray_config_ppo.json.template) and 
[DQN](ray_config_dqn.json.template) are provided.

### Experiment configuration file

The `exp_config.json` includes parameters related to the training experiment 
(i.e., the Algorithm to use or the stopping criteria) and/or the configuration 
of the Ray `Tuner` for automatic hyperparameter tuning.

The only **mandatory parameters** are `algorithm`, which corresponds to the 
name of the RL Algorithm to use, and `stopping_criteria`, which is a dictionary 
used to list the (possibly, multiple) stopping criteria to be considered 
during the training. The available termination conditions are:
- `max_iterations`: the maximum number of training iterations.

Additional (optional) parameters are:
- `logdir`: the base directory where all the experiments outputs should be 
saved. If, e.g., `OUTPUT` is provided as `logdir`, a subdirectory is created 
there with the following name: `f"{algo}_{environment}_{now}"`, where `algo` 
is the name of the Ray `Algorithm`, `environment` is the name of the chosen 
`Environment` and `now` is given by 
`datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')`. The default base result 
directory if no value is provided here is `~/ray_results`.
- `from_checkpoint`: path to the directory where the checkpoint to be 
restored is saved. If this is provided, further information related to the 
Environment or the Ray Algorithm configuration files are neglected.
- `env_config_file`: path to the `env_config.json` file described 
[above](#environment-configuration-file). This parameter is **mandatory** if 
no previous checkpoint is provided.
- `ray_config_file`: path to the `ray_config.json` file described 
[above](#ray-algorithm-configuration-file).
- `evaluation_interval`: after how many iterations the evaluation should be 
performed. **Important note:** one evaluation step is always performed at the 
end of the training loop, even if no parameter is provided here.
- `checkpoint_interval`: after how many iterations an algorithm checkpoint 
should be saved. **Important note:** one checkpoint is always saved at the 
end of the training loop, even if no parameter is provided here.
- `plot_interval`: TBA

Example: 

```
{
  "algorithm": "PPO",
  "env_config_file": "config_files/env_config.json",
  "ray_config_file": "config_files/ray_config.json",
  "logdir": "OUTPUT",
  "evaluation_interval": 5,
  "checkpoint_interval": 5,
  "stopping_criteria": {
    "max_iterations": 10
  }
}
```

#### Configure experiment logging

To configure the RL4CC `Logger` verbosity level and the output/error stream 
it considers, a `logger` sub-dictionary should be added, including the 
following parameters:
- `verbosity`: an integer between 0 (minimum verbosity) and 3 (maximum 
verbosity), which controls how many messages are printed during the 
experiment. Indeed, while warnings and errors are always printed, generic 
information are printed only if the verbosity is higher than the corresponding 
message level.
- `file_streams`: `True` to log on files instead of using `sys.stdout` and 
`sys.stderr`.

#### Configure hyperparameter tuning

To configure a Ray `Tuner`, a `tuner` sub-dictionary should be added 
in `exp_config.json`. It is used to specify, e.g., the number of times to 
sample from the hyperparameter space and the policy for fault tolerance when 
resuming a stopped experiment from the existing state. In particular, the 
`resume_errored` and `restart_errored` fields are related to the possibility 
of resuming or restarting an experiment left in the `ERRORED` state, 
respectively. The `resume_unfinished` field is related to the possibilty of 
resuming an experiment left in the `RUNNING` state. 

**Note:** experiments left in the `TERMINATED` state cannot be resumed: you 
have to start a new experiment from scratch if you want to test new parameters.

Concerning the `num_tune_samples` parameter, if this is -1, (virtually) 
infinite samples are generated until a stopping condition is met.

Example of `tuner` sub-dictionary:

```
{
  "algorithm": "...",
  "stopping_criteria": {...},
  ...,
  "tuner": {
    "num_tune_samples": 2,
    "resume_errored": true,
    "restart_errored": false,
    "resume_unfinished": true
  }
}
```
