## Expected structure of the configuration

Each experiment is controlled by a base configuration, a dictionary, called
`exp_config`. It contains information about the experiment to run (e.g., the
name of the algorithm to use, whether to start from an existing checkpoint,
etc.).

RL4CC experiment classes (`BaseExperiment`, `TrainingExperiment` and
`TuningExperiment`) can read the `exp_config` in two ways:

1. From a JSON file (like `exp_config.json`) using the `exp_config_file`
   parameter,
2. From a dictionary using the `exp_config` parameter.

> [!WARNING]
> You must specify one of the two parameters and they cannot be specified at the
> same time.

If the experiment should start from scratch (i.e., no previous checkpoints
are available), the configuration is completed by two (or three) additional
files, which provide information about the `Environment`, the Ray `Algorithm`
and, possibly, the `Tuner` configuration.

The corresponding structure is detailed in the following.

### `Environment` configuration

The `env_config` must include **four** mandatory parameters, which are related
to Environment name and the simulation time management within it.

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

### Ray `Algorithm` configuration

The `ray_config` file must include parameters related to the definition of a Ray
[`AlgorithmConfig`](https://docs.ray.io/en/latest/rllib/rllib-training.html#configuring-rllib-algorithms)
object.

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

> [!WARNING]
> To simplify the management of some parameters related to the experience
> sampling and training, RL4CC offers the possibility of setting higher-level
> _suggested_ keywords instead of directly using the ones defined in Ray. These
> are:
>
> - In the `rollouts` section:
>
>   - `duration_unit`: it can take the value `episodes`, if rollout workers
>     should collect entire episodes during the experience sampling phase, or
>     `timesteps`, if episodes can be truncated during the experience sampling;
>
>   - `duration_per_worker`: how many episodes/steps should be collected by each
>     rollout worker.
>
> - In the `training` section:
>
>   - `batch_size`: dimension of each batch extracted from the collected
>     experience (or the replay buffer, if defined) during the training phase;
>
>   - `num_train_batches`: how many batches should be trained in each iteration.
>
> - in the `resources` section:
>
>   - `num_gpus_master`: number of GPUs assigned to the master node;
>
>   - `num_cpus_master`: number of CPUs assigned to the master node.
>
> - in the `evaluation` section:
>   - `evaluation_duration_per_worker`: how many episodes/steps should be
>     collected by each evaluation worker.
>
> These keywords mask a lower-level management performed by Ray, where different
> algorithms use different parameters to control the same elements. An
> **expert** user is free to set directly the Ray _protected_ keywords, but the
> two approaches cannot be mixed.

> [!WARNING]
> There are few elements that, differently from what is explained in the Ray
> documentation **should not** be managed through `ray_config`. These are:
>
> - `env` and `env_config`, from the `environment` parameters group;
>
> - `evaluation_interval`, from the `evaluation` parameters group;
>
> - `logdir`, from the `logger_config` dictionary in the `debugging` parameters
>   group.
>
> In particular, `env` and `env_config` are indirectly controlled through the
> [`env_config`](#environment-configuration) configuration, while
> `evaluation_interval` and `logdir` are set from the [experiment configuration
> file](#experiment-configuration).

> [!NOTE]
> In the `callbacks` section, the `callbacks_class` parameter should correspond
> to the path to the callbacks class as it would be reported while importing the
> module (e.g., `"callbacks.base_callbacks.BaseCallbacks"`).

Sample `ray_config.json` files for [PPO](ray_config_ppo.json.template) and
[DQN](ray_config_dqn.json.template) are provided.

#### How to use custom Policy models

Ray supports the use of Torch and TF models using the ModelV2 implementation.

One can easily implement their own custom network by extending either the
[Torch](../models/base_torch_model.py) or the
[Tensorflow](../models/base_tf_model.py) Base models, and then registering in
Ray `ModelCatalog`. We provide two sample custom models as starting points,
and the instructions to register them can be seen in the
[models initialization file](../models/__init__.py).

To actually indicate the use of a custom model after building and registering
it, one must specify the **name** of the model (as it was registered in the
catalog) and the associated custom model config dictionary in the `training`
section of the Ray configuration file.

For example, to use the provided Torch-based
[CustomTorchModel](../models/custom_torch_model.py), which is registered
under the name `custom_torch_model`, provide in the ray_config.json file,
under the `model` dictionary in the `training` section, the following
information:

```
.
.
"framework": "torch",
.
.
"training": {
  .
  .
  "model": {
    "custom_model": "custom_torch_model",
    "custom_model_config": {
      "seed": 123,
      "fun_layers": ["ReLU", "ReLU", "ReLU"],
      "dropout": true,
      "dropout_list": [0.02, 0, 0],
      "n_features": [128, 128, 64]
    }
  },
  .
  .
}
.
.
```

Examples are reported for both the PPO and DQN algorithms in the the
corresponding template files.

> [!WARNING]
> The framework the model is based on **must** match the framework passed in the
> `ray_config`, otherwise this will result in runtime errors.

### Configure hyperparameter tuning

To run hyperparameter tuning, the user should:

1. provide a `tune_config` (including parameters related to the configuration of
   the `Tuner` object), and
2. adapt the `ray_config` file to properly define the search space.

Details are provided in the following.

#### `Tuner` configuration

The `tune_config` must include **three** mandatory parameter, which are related
to the number of tune trials and the identification of the best result.

These are:

- `num_tune_trials`: the number of tuning trials (possibly run in parallel, if
  the cluster resources are enough to do so). These trials will sample from the
  Tune search space (defined according to the elements in the `ray_config`, as
  detailed in [the next section](#configuring-the-search-space-for-parameters)).
  **Note that,** if the `num_tune_trials` parameter is -1, (virtually) infinite
  samples are generated until a stopping condition is met.
- `metric`: the metric used to evaluate the performance of a given set of
  parameters in a trial.
- `mode`: the mode on which the values returned by the metric are evaluated.
  For instance, setting this parameter to `max` when considering an
  `epsiode_reward_mean` metric, will place the trial that returns the highest
  numerical value of the mean reward as the best trials.

Additional (optional) parameters are:

- `search_algorithm`: a search algorithm, as specified in the
  [tune.search](https://docs.ray.io/en/latest/tune/api/suggestion.html#hyperopt-tune-search-hyperopt-hyperoptsearch) page
- `scheduler`: a scheduler, as specified in [tune.schedulers](https://docs.ray.io/en/latest/tune/api/schedulers.html#tune-scheduler-hyperband) page
- instructions on how to proceed when restoring a `Tuner` from an existing
  checkpoint. These include:
  - the `resume_errored` and `restart_errored` fields, which are related to
    the possibility of resuming or restarting an experiment left in the
    `ERRORED` state, respectively. By default, both are `False`.
  - the `resume_unfinished` field, related to the possibilty of resuming an
    experiment left in the `RUNNING` state. By default, it is `True`.
- `progress_reporter`: the configuration of a custom `ProgressReporter`, 
  including:
  - `progress_reporter_class`: as for the callbacks, it should correspond
  to the path to the progress reporter class as it would be reported while 
  importing the module (e.g., 
  `"callbacks.base_tune_progress_reporter.BaseProgressReporter"`).
  - `progress_reporter_config`: a dictionary of parameters to be passed 
  as keyword arguments to the `ProgressReporter` constructor. Note that, since 
  the default progress reporters are not designed to ignore unwanted 
  keyword arguments, an error will be thrown if the provided parameter is 
  not an expected one.

> [!NOTE]
> Currently, only the `HyperOpt` search algorithm and the `ASHAScheduler` are
> implemented.

> [!WARNING]
> Experiments left in the `TERMINATED`
> state cannot be resumed: you have to start a new experiment from scratch if you
> want to test new parameters or change other configuration terms as the metric,
> mode or number of tune trials.

The provided `BaseProgressReporter` extends the 
[`TuneReporterBase`](https://github.com/ray-project/ray/blob/master/python/ray/tune/progress_reporter.py) 
class, which accepts the following parameters:
- `metric_columns`: Names of metrics to include in progress table. If this is 
  a dict, the keys should be metric names and the values should be the 
  displayed names. If this is a list, the metric name is used directly.
- `parameter_columns`: Names of parameters to include in progress table. If 
  this is a dict, the keys should be parameter names and the values should be 
  the displayed names. If this is a list, the parameter name is used directly. 
  If empty, defaults to all available parameters.
- `max_progress_rows`: Maximum number of rows to print in the progress table. 
  The progress table describes the progress of each trial. Defaults to 20.
- `max_error_rows`: Maximum number of rows to print in the error table. The 
  error table lists the error file, if any, corresponding to each trial. 
  Defaults to 20.
- `max_column_length`: Maximum column length (in characters). Column headers 
  and values longer than this will be abbreviated.
- `max_report_frequency`: Maximum report frequency in seconds. Defaults to 5s.
- `infer_limit`: Maximum number of metrics to automatically infer from 
  tune results.
- `print_intermediate_tables`: Print intermediate result tables. If `None` 
  (default), will be set to `True` for verbosity levels above 3, otherwise 
  `False`. If `True`, intermediate tables will be printed with experiment 
  progress. If `False`, tables will only be printed at then end of the tuning 
  run for verbosity levels greater than 2.
- `metric`: Metric used to determine best current trial.
- `mode`: One of `[min, max]`. Determines whether objective is minimizing or 
  maximizing the metric attribute.
- `sort_by_metric`: Sort terminated trials by metric in the intermediate 
  table. Defaults to `False`.

Moreover, it accepts an additional parameter, named `progress_file`, denoting 
the path to the file where the experiment progress should be periodically 
logged. You can provide either the complete path to a file of your choice or 
`None`; in the second case, the progress file will default to 
`exp_progress.json` (see the example below).

Sample configuration (with a custom `ProgressReporter` logging on 
`exp_progress.json`):

```
{
  "num_tune_trials": 10,
  "metric": "episode_reward_mean",
  "mode": "max"
  "search_algorithm": {
    "hyperopt_search": {
      "metric": "episode_reward_mean",
      "mode": "max"
    }
  },
  "scheduler": {
    "asha_scheduler": {
      "grace_period": 10,
      "reduction_factor": 3,
      "brackets": 1
    }
  },
  "progress_reporter": {
    "progress_reporter_class": "callbacks.base_tune_progress_reporter.BaseProgressReporter",
    "progress_reporter_config": {
      "progress_file": null
    }
  }
}
```

> [!WARNING]
> Due to an ongoing [issue](https://github.com/ray-project/ray/issues/38202) 
> with recent Ray versions, you must set the environment variable 
> `RAY_AIR_NEW_OUTPUT=0` to be able to use custom `ProgressReporter`s.

> [!NOTE]
> Choose a large-enough verbosity level (i.e., greater than 0) in the 
> experiment configuration to ensure that the `ProgressReporter` works as 
> expected.

#### Configuring the `search space` for parameters

The Tuner configuration `tune_config` described in the section
[above](#tuner-configuration) is responsible for the behaviour of the `Tuner`
and how it handles the `Trials` running in parallel. However, to actually
fine-tune algorithm parameters, the user should define the search space by
suitably adapting the `ray_config`.

This can be simply done by providing the values of each parameter to be tuned as
a `tune.**search_space` string. For example, if the learning rate for the PPO
algorithm is to be tuned, locate the corresponding parameter (`lr`, in the
`training` section) in the `ray_config` file and set it as:

```
"training": {
  "gamma": 0.99,
  "grad_clip": null,
  "grad_clip_by": "global_norm",
  "lr": "tune.loguniform(1e-4, 1e-1)",
  .
  .
  .
}
```

In the above example, the `Tuner` will sample `num_tune_trials` (present in the
`tune_config` file) trials; each trial will run for the specified number of
`max_iterations` (present in the `exp_config` file, as discussed in [the
following](#experiment-configuration)), considering a different `lr` value
sampled from the `loguniform` distribution over the range of `(1e-4, 1e-1)`.

For more about tune search spaces see
[the relative documentation](https://docs.ray.io/en/latest/tune/api/search_space.html).

> [!WARNING]
> When the user tries to start a new tuning experiment without specifying the
> path to a `tune_config.json` file in the `exp_config.json` file, the execution
> will be interrupted prompting the user to provide them.

### Experiment configuration

The `exp_config` includes parameters related to the training and/or
hyperparameter tuning experiment (e.g., the algorithm to use, the stopping
criteria, etc.).

The **mandatory parameters** vary depending on whether a simple training or
a hyperparameter tuning experiment is defined. Note, however, that:

- the `algorithm` parameter, which corresponds to the name of the RL Algorithm
  to use, is always mandatory.
- the path to the environment configuration file (and for the tuning
  configuration file in the case of a `TuningExperiment`) is always mandatory
  if the experiment is not restored from an existing checkpoint.
- the `stopping_criteria` dictionary must be provided list the (possibly,
  multiple) stopping criteria to be considered during the training. The only
  exception is when a `TuningExperiment` is restored from a checkpoint, when the
  stopping criterion is copied from the previous run. The available termination
  conditions are:
  - `max_iterations`: the maximum number of training iterations.

Additional parameters are:

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

> [!WARNING] 
> In the case of `TuningExperiment`s, the path is the path to the general 
> tuning experiment outputs folder within `logdir`, not the path to a 
> specific checkpoint directory.

- `env_config_file`: path to the `env_config.json` file described
  [above](#environment-configuration).
- `env_config`: dictionary containing the environment configuration described
  [above](#environment-configuration).
- `ray_config_file`: path to the `ray_config.json` file described
  [above](#ray-algorithm-configuration).
- `ray_config`: dictionary containing the Ray configuration described
  [above](#ray-algorithm-configuration).
- `tune_config_file`: path to the `tune_config.json` file described
  [above](#tuner-configuration).
- `tune_config`: dictionary containing the tuner configuration described
  [above](#tuner-configuration).
- `evaluation_interval`: after how many iterations the evaluation should be
  performed. **Important note:** one evaluation step is always performed at the
  end of the training loop, even if no parameter is provided here.
- `checkpoint_interval`: after how many iterations an algorithm checkpoint
  should be saved. **Important note:** one checkpoint is always saved at the
  end of the training loop, even if no parameter is provided here.
- `plot_interval`: TBA

> [!WARNING]
> If no previously checkpoint is provided, you **must** specify either
> `env_config_file` or `env_config` but not both. The same applies to
> Ray config (`ray_config_file` and `ray_config`) and tuner configuration
> (`tune_config_file` and `tune_config`),

Example (for a training experiment):

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

Example (for a hyperparameter tuning experiment):

```
{
  "algorithm": "PPO",
  "env_config_file": "config_files/env_config.json",
  "ray_config_file": "config_files/ray_config.json",
  "tune_config_file": "config_files/tune_config.json",
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
