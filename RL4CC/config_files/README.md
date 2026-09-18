## Expected structure of the configuration

Each experiment is controlled by a base configuration, a dictionary, called
`exp_config`. It contains information about the experiment to run (e.g., the
name of the algorithm to use, whether to start from an existing checkpoint,
etc.).

RL4CC experiment classes (`BaseExperiment`, `TrainingExperiment` -- with 
its two variants for [Federated RL](../experiments/federated_train.py) and 
[Gossip RL](../experiments/gossip_train.py) -- and
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

The corresponding structure is detailed in the following sections:
- [How to configure the environment](#environment-configuration)
  - [Multi-agent environments](#multi-agent-environments)
  - [Environments for gossip RL](#gossip-rl)
- [How to configure the RL algorithm](#ray-algorithm-configuration)
  - [Custom policy models](#how-to-use-custom-policy-models)
- [How to configure hyperparameter tuning](#configure-hyperparameter-tuning)
  - [The tuner](#tuner-configuration)
  - [The search space](#configuring-the-search-space-for-parameters)
- [How to configure the experiment](#experiment-configuration)
  - [Federated RL experiments](#federated-rl-experiments)
  - [Gossip RL experiments](#gossip-rl-experiments)
  - [Experiment logging](#configure-experiment-logging)


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

> [!NOTE] 
> These parameters should be set even in a non-episodic environment. Use 
> `max_time` to represent the Environment horizon.

#### Multi-agent environments

To create multi-agent environments, an additional parameter must be set in 
the environment configuration dictionary, namely `agents`, providing a list 
of agents names or IDs. As an example:

```
{
  "env_name": "BaseMultiAgentEnvironment",
  "min_time": 0,
  "max_time": 3600,
  "time_step": 10,
  "agents": ["agent_0", "agent_1", "agent_2"]
}
```

#### Gossip RL

For gossip experiments, multi-agent environments must also define a 
neighborhood  mapping in  env_config , because the gossip training class 
reads that structure to determine which agents can exchange models during 
each round.

A typical example (for a fully-connected network) is:

```
{
  "env_name": "BaseMultiAgentEnvironment",
  "min_time": 0,
  "max_time": 3600,
  "time_step": 10,
  "agents": ["agent_0", "agent_1", "agent_2"],
  "neighborhood": {
    "agent_0": ["agent_1", "agent_2"],
    "agent_1": ["agent_0", "agent_2"],
    "agent_2": ["agent_0", "agent_1"]
  }
}
```

### How to configure the learner

The `learner_config` file contains the parameters used to initialize the
learning algorithm. RL4CC supports two algorithm backends:

- **Ray RLlib**, used by algorithms such as `PPO`, `DQN`, `SAC`, and `MAPPO`;
- **MORL-Baselines**, currently used by `EnvelopeQLearning`.

The backend is selected automatically from the algorithm registered in
[`AGfactory`](../algorithms/generators_factory.py). Therefore, the
`learner_config` has the same high-level structure in both cases, while the
parameters inside the sections are interpreted according to the selected
backend.

> [!IMPORTANT]
> The `learner_config` is intentionally organized around the same three
> high-level sections for both backends:
>
> - `rollouts`: how much experience is collected for each training iteration;
> - `evaluation`: how evaluation is configured;
> - `training`: algorithm-specific training parameters.
>
> This common structure does **not** mean that the internal parameters have the
> same meaning or names in RLlib and MORL-Baselines. RL4CC's algorithm
> generators translate a set of common, higher-level keys into the parameters
> expected by the selected implementation. Parameters not translated by RL4CC
> are passed to the underlying library and must therefore follow that library's
> documentation.

#### Common configuration keys

The following keys are interpreted by both RLlib and MORL-Baselines generators.

##### `rollouts`

- `duration_unit`: unit used to express the amount of experience collected
  during one training iteration. It can be `timesteps` or `episodes`.
  With `timesteps`, `duration_per_worker` is interpreted directly as a number
  of environment steps. With `episodes`, RL4CC uses `min_time`, `max_time` and
  `time_step` from `env_config` to determine the number of steps in one
  episode.
- `duration_per_worker`: amount of experience collected per rollout worker,
  expressed in the unit selected by `duration_unit`.
- `num_rollout_workers`: number of rollout workers. RLlib can use multiple
  workers. The current MORL-Baselines integration supports **only one** rollout
  worker, so this value must be `1` (or omitted, in which case the default is
  `1`).

>[!WARNING]
> When implementing multi-agent RL, consider that Ray RLLib allows to decide 
> if the rollout steps should be counted as *agents' steps* or 
> *environment's steps*, based on the value of the `count_steps_by` parameter 
> to be specified in an additional 
> [`multi_agent` section](https://docs.ray.io/en/releases-2.20.0/?utm_source=ray_io&utm_medium=website&utm_campaign=nav). 
> The two supported values are: "env_steps", which counts each time the env 
> is "stepped" (no matter how many multi-agent actions are passed/how many 
> multi-agent observations have been returned in the previous step), and 
> "agent_steps", which counts each individual agent step as one step.

##### `evaluation`

- `evaluation_duration_unit`: unit used for the evaluation duration;
  `timesteps` or `episodes` as for the `rollouts` above.
- `evaluation_duration_per_worker`: amount of experience collected by each
  evaluation worker, expressed in `evaluation_duration_unit`.
- `evaluation_num_workers`: number of evaluation workers. The current
  MORL-Baselines integration supports **only one** evaluation worker. 
- `evaluation_config`: additional configuration for the evaluation environment.
  RL4CC starts from `env_config` and applies the values in this dictionary as
  overrides. In the MORL-Baselines backend, these values are used to create
  the separate evaluation environment.

>[!NOTE]
> The evaluation interval itself is **not** configured in `learner_config`.
> Set it through `evaluation_interval` in `exp_config`, as described in
> [Experiment configuration](#experiment-configuration). RL4CC converts that
> value to the corresponding backend-specific evaluation frequency.

>[!CAUTION]
> Due to a known issue with the currently supported RLlib API stack, 
> configuring `evaluation_num_workers` can result in an inconsistent number 
> of evaluation episodes. It is therefore recommended to omit this key 
> from the configuration entirely. 
> See [`ray_issues_to_track.md`](../../ray_issues_to_track.md).

##### `training`

- `batch_size`: size of each training batch. The underlying parameter to which
  this is translated depends on the algorithm.
- `num_train_batches`: number of training batches/gradient-update groups
  performed in one RL4CC training iteration. The exact underlying parameter
  used by the algorithm is backend-specific.

The two generators also provide common translations for several training
parameters, including `lr` (learning rate), `grad_clip` (gradient clipping),
and `hiddens` (network architecture), but their final interpretation is
determined by the underlying algorithm implementation. In particular, the
MORL-Baselines generator translates these to `learning_rate`,
`max_grad_norm`, and `net_arch`, respectively.

> [!CAUTION]
> `batch_size` and `num_train_batches` are deliberately expressed using
> RL4CC's common terminology. They are not necessarily passed to the
> underlying library under the same names. Do not assume that an RLlib
> parameter can be copied verbatim into a MORL-Baselines configuration, or
> vice versa.

#### Ray RLlib learner configuration

For a Ray RLlib algorithm, the `learner_config` parameters are used to build
the corresponding Ray `AlgorithmConfig`.

The sections commonly used by RLlib are:

- `framework`, for the deep-learning framework;
- `rollouts`, for experience collection;
- `evaluation`, for evaluation;
- `resources`, for the resources allocated to the algorithm and rollout workers;
- `training`, for common and algorithm-specific training parameters;
- `exploration`, for exploration behaviour;
- `callbacks`, for RL4CC/RLlib callbacks;
- `debugging`, for Ray/RLlib debugging and logging options.

A comprehensive description of the RLlib parameters is available in the
[Ray RLlib configuration documentation](https://docs.ray.io/en/releases-2.20.0/rllib/rllib-training.html#configuring-rllib-algorithms).

RL4CC additionally provides the following higher-level keys:

- `num_gpus_master`: number of GPUs assigned to the master/local worker;
- `num_cpus_master`: number of CPUs assigned to the master/local worker.

These are translated by RL4CC to the corresponding RLlib resource
configuration. Other resource parameters should follow the RLlib
documentation.

> [!CAUTION]
> Some RLlib parameters are managed by RL4CC and should not be set directly
> in `learner_config`:
>
> - `env` and `env_config`, which are derived from the
>   [`env_config`](#environment-configuration);
> - `evaluation_interval`, which is taken from `exp_config`;
> - `logdir` inside `logger_config`, which is set from the experiment output
>   directory.
>
> RL4CC's higher-level suggested keys and the corresponding lower-level
> protected RLlib keys should not be mixed.

For RLlib algorithms, sample `learner_config.json` files are provided for
[PPO](learner_config_ppo.json.template), [DQN](learner_config_dqn.json.template),
and [MAPPO](learner_config_mappo.json.template).

##### Custom policy models

Ray supports the use of Torch and TF models using the ModelV2 implementation.

One can implement a custom network by extending either the
[Torch](../models/base_torch_model.py) or the
[Tensorflow](../models/base_tf_model.py) Base model, and then registering it in
Ray's `ModelCatalog`. RL4CC provides sample custom models and the registration
instructions in the [models initialization file](../models/__init__.py).

To select a registered custom model, specify its registered name and the
associated custom model configuration in the `model` dictionary inside the
`training` section.

For example, for the provided Torch-based
[`CustomTorchModel`](../models/custom_torch_model.py), registered as
`custom_torch_model`:

```json
{
  "framework": "torch",
  "training": {
    "model": {
      "custom_model": "custom_torch_model",
      "custom_model_config": {
        "seed": 123,
        "fun_layers": ["ReLU", "ReLU", "ReLU"],
        "dropout": true,
        "dropout_list": [0.02, 0, 0],
        "n_features": [128, 128, 64]
      }
    }
  }
}
```

Examples are reported in the PPO and DQN template files.

> [!WARNING]
> The framework used by the custom model must match the `framework` selected
> in `learner_config`, otherwise runtime errors can occur.

A notable custom policy is the
[centralized critic model](../models/centralized_critic_model.py) used by
MAPPO. It is registered in RL4CC as `centralizedcritic`; see the
[MAPPO configuration template](learner_config_mappo.json.template).

#### MORL-Baselines learner configuration

For a MORL-Baselines algorithm, RL4CC instantiates the algorithm directly
through its MORL-specific generator. The current integration registers
`EnvelopeQLearning`, implemented using MORL-Baselines' `Envelope` algorithm.

MORL-Baselines follows the
[MO-Gymnasium API](https://github.com/Farama-Foundation/MO-Gymnasium) and
provides its own algorithm-specific parameters. See the
[MORL-Baselines documentation](https://lucasalegre.github.io/morl-baselines/)
for the general API and the
[Envelope Q-Learning documentation](https://lucasalegre.github.io/morl-baselines/algos/multi_policy/envelope/)
for the algorithm-specific parameters.

The important distinction is that RL4CC keeps the same `rollouts`,
`evaluation`, and `training` organization, but the MORL generator translates
those sections into MORL-Baselines constructor/training parameters.

For example, a minimal Envelope Q-Learning configuration can be written as:

```json
{
  "rollouts": {
    "duration_unit": "timesteps",
    "duration_per_worker": 360
  },
  "training": {
    "lr": 3e-4,
    "gamma": 0.98,
    "batch_size": 64,
    "num_train_batches": 1,
    "hiddens": [256, 256, 256, 256],
    "replay_buffer_config": {
      "capacity": 2000000
    },
    "ref_point": [1.0],
    "seed": 4850
  },
  "exploration": {
    "initial_epsilon": 1.0,
    "final_epsilon": 0.05,
    "epsilon_decay_steps": 50000
  }
}
```

A complete working example is available in
[`learner_config_envelopeqlearning.json.template`](learner_config_envelopeqlearning.json.template).

For MORL-Baselines, the following RL4CC translations are particularly
important:

- `rollouts.duration_unit` + `rollouts.duration_per_worker` become the total
  training horizon (`total_timesteps`, and `total_episodes` when the duration
  is specified in episodes).
- `training.lr` becomes `learning_rate`.
- `training.grad_clip` becomes `max_grad_norm`.
- `training.hiddens` becomes `net_arch`.
- `training.num_train_batches` becomes `gradient_updates`.
- `training.num_steps_sampled_before_learning_starts` becomes
  `learning_starts`.
- `training.replay_buffer_config.capacity` becomes `buffer_size`.
- `training.replay_buffer_config.prioritized_replay_alpha` enables prioritized
  experience replay and becomes `per_alpha`.
- `training.target_network_update_freq` becomes `target_net_update_freq`.
- `training.ref_point` is converted to a NumPy array and is used by RL4CC
  when computing multi-objective evaluation metrics.

The MORL generator also maps the `exploration` section to the corresponding
epsilon parameters used by Envelope Q-Learning. If `exploration.explore` is
set to `false`, epsilon exploration is disabled.

MORL-Baselines-specific parameters that are not translated by RL4CC are passed
to the `Envelope` implementation using the parameter names expected by
MORL-Baselines. Consult the
[Envelope Q-Learning documentation](https://lucasalegre.github.io/morl-baselines/algos/multi_policy/envelope/)
when adding or changing such parameters.

> [!WARNING]
> The current MORL-Baselines integration supports only one rollout worker and
> one evaluation worker. Setting `num_rollout_workers` or
> `evaluation_num_workers` to a value other than `1` is rejected.

> [!NOTE]
> `evaluation_interval` is still controlled by `exp_config`, exactly as for
> RLlib. RL4CC converts it to the MORL-Baselines evaluation frequency.
> `evaluation_duration_per_worker` is converted to a number of evaluation
> episodes; when it is expressed in timesteps, RL4CC derives the number of
> episodes from the episode length specified by `env_config`.

> [!NOTE]
> MORL-Baselines algorithms are multi-objective algorithms and therefore
> require a multi-objective environment. The environment must follow the
> MO-Gymnasium interface and expose the reward vector expected by the selected
> MORL-Baselines algorithm. Any custom multi-objective environment to be 
> used with RL4CC should inherit from the provided 
> [`BaseMultiObjectiveEnvironment`](../environment/base_multiobjective_environment.py)

### Configure hyperparameter tuning

To run hyperparameter tuning, the user should:

1. provide a `tune_config` (including parameters related to the configuration of
   the `Tuner` object), and
2. adapt the `learner_config` file to properly define the search space.

Details are provided in the following.

#### `Tuner` configuration

The `tune_config` must include **three** mandatory parameter, which are related
to the number of tune trials and the identification of the best result.

These are:

- `num_tune_trials`: the number of tuning trials (possibly run in parallel, if
  the cluster resources are enough to do so). These trials will sample from the
  Tune search space (defined according to the elements in the `learner_config`, as
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
  [tune.search](https://docs.ray.io/en/releases-2.20.0/tune/api/suggestion.html#tune-search-algorithms-tune-search) page
- `scheduler`: a scheduler, as specified in [tune.schedulers](https://docs.ray.io/en/releases-2.20.0/tune/api/schedulers.html) page
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
  `"RL4CC.log_and_report.base_tune_progress_reporter.BaseProgressReporter"`).
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
    "progress_reporter_class": "RL4CC.log_and_report.base_tune_progress_reporter.BaseProgressReporter",
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
suitably adapting the `learner_config`.

This can be simply done by providing the values of each parameter to be tuned as
a `tune.**search_space` string. For example, if the learning rate for the PPO
algorithm is to be tuned, locate the corresponding parameter (`lr`, in the
`training` section) in the `learner_config` file and set it as:

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
[the relative documentation](https://docs.ray.io/en/releases-2.20.0/tune/api/search_space.html).

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
  stopping criterion is copied from the previous run. For standard training 
  and hyperparameter tuning experiments, the available termination condition 
  `max_iterations`, which denotes the maximum number of training iterations. 
  Federated and gossip RL training experiments require both `max_iterations` 
  and `max_federation_rounds`: the federated stopping logic checks both the 
  local training horizon (`max_iterations` to be run in each round) and the 
  number of federation rounds to consider.

Additional parameters are:

- `logdir`: the base directory where all the experiments outputs should be
  saved. If, e.g., `OUTPUT` is provided as `logdir`, a subdirectory is created
  there with the following name: `f"{algo}_{environment}_{now}"`, where `algo`
  is the name of the Ray `Algorithm`, `environment` is the name of the chosen
  `Environment` and `now` is given by
  `datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')`. The default base result
  directory if no value is provided here is `~/ray_results`.
- `from_checkpoint`: path to the checkpoint to be restored. When using 
  Ray-based algorithms, if this is provided, further information related to 
  the environment or the learner configuration are neglected. 

> [!WARNING] 
> In the case of `TuningExperiment`, the `from_checkpoint` path is the path to 
> the general tuning experiment outputs folder within `logdir`, not the path 
> to a specific checkpoint directory. In the case of a `TrainingExperiment`, 
> it is the path to a checkpoint directory when working with Ray-based 
> algorithms, and the path to the checkpoint .tar file when working with 
> algorithms based on MORL-Baselines

- `env_config_file`: path to the `env_config.json` file described
  [above](#environment-configuration).
- `env_config`: dictionary containing the environment configuration described
  [above](#environment-configuration).
- `learner_config_file`: path to the `learner_config.json` file described
  [above](#ray-algorithm-configuration).
- `learner_config`: dictionary containing the Ray configuration described
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
- `save_manual_checkpoints`: if `True`, saves checkpoints by dumping the whole 
  `Algorithm` state instead of relying on Ray RLLib checkpointing mechanism. 
  This solves some ongoing issues with performance drop after reloading.
- `plot_interval`: TBA

> [!WARNING]
> If no previously checkpoint is provided, you **must** specify either
> `env_config_file` or `env_config` but not both. The same applies to
> Ray config (`learner_config_file` and `learner_config`) and tuner configuration
> (`tune_config_file` and `tune_config`). For multi-objective experiments, 
> the environment and learner configurations must be provided regardless 
> the fact that the experiment re-starts from an existing checkpoint.

Example (for a training experiment):

```
{
  "algorithm": "PPO",
  "env_config_file": "config_files/env_config.json",
  "learner_config_file": "config_files/learner_config.json",
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
  "learner_config_file": "config_files/learner_config.json",
  "tune_config_file": "config_files/tune_config.json",
  "logdir": "OUTPUT",
  "evaluation_interval": 5,
  "checkpoint_interval": 5,
  "stopping_criteria": {
    "max_iterations": 10
  }
}
```

#### Federated RL experiments

The following additional, non-mandatory exp_config keys are supported by 
federated training:
- `n_train_iterations_before_federation_starts`: extra number of local 
  training iterations to execute before starting the first federation round.
- `networks_to_aggregate`: list of layer-name prefixes that identify which 
  subnetworks can be aggregated. Its default value is `["all"]`, meaning that 
  all subnetworks weights are averaged in each round (unless included in the 
  list of private layers).
- `private_layers`: dual to the former, list of layer-name prefixes that must 
  remain local and are never aggregated.

> [!WARNING]
> `networks_to_aggregate` and `private_layers` cannot overlap. An error is 
> raised if the same prefix appears in both lists.

A minimal example of experiment configuration for federated training is:

```
{
  "algorithm": "PPO",
  "env_config_file": "config_files/env_config.json",
  "learner_config_file": "config_files/learner_config.json",
  "logdir": "OUTPUT",
  "evaluation_interval": 5,
  "checkpoint_interval": 5,
  "networks_to_aggregate": ["policy", "encoder"],
  "private_layers": ["value_head"],
  "n_train_iterations_before_federation_starts": 10,
  "stopping_criteria": {
    "max_iterations": 5,
    "max_federation_rounds": 20
  }
}
```

In the previous example, all layers whose names start with `policy` or 
`encoder` are candidates for aggregation, while the `value_head` layers are 
kept local to each agent. The first federation round performs 10 + 5 local 
iterations because the warm-up term is added only before the first aggregation 
step, while later rounds run 5 training iterations each.

#### Gossip RL experiments

[As already mentioned](#environment-configuration), the environment 
configuration for gossip experiments must include a `neighborhood` dictionary 
mapping each agent to the list of agents it may sample from for aggregation. 

The gossip experiment adds the following non-mandatory configuration fields 
on top of [the federated ones](#federated-rl-experiments):
- `n_models_to_share`: by default, 1. Denotes the number of neighbors sampled 
  (without replacement) for aggregation by each agent at each round.
- `exp_seed`: seed used to initialize the NumPy random generator that selects 
  the neighbors to ensure reproducibility.

> [!WARNING]
> During validation, the class reads this structure from `env_config`, reads 
> `n_models_to_share` from `exp_config`, and checks that the requested number 
> of sampled neighbors does not exceed the degree of any agent. 

A minimal example is:

```
{
  "algorithm": "PPO",
  "env_config_file": "config_files/env_config_gossip.json",
  "learner_config_file": "config_files/learner_config.json",
  "logdir": "OUTPUT",
  "evaluation_interval": 5,
  "checkpoint_interval": 5,
  "networks_to_aggregate": ["policy"],
  "private_layers": ["critic"],
  "n_models_to_share": 1,
  "exp_seed": 4850,
  "n_train_iterations_before_federation_starts": 10,
  "stopping_criteria": {
    "max_iterations": 5,
    "max_federation_rounds": 20
  }
}
```

assuming the corresponding neighborhood defined in the environment 
configuraiton:

```
{
  .
  .
  .
  "agents": ["agent_0", "agent_1", "agent_2", "agent_3"],
  "neighborhood": {
    "agent_0": ["agent_1", "agent_2"],
    "agent_1": ["agent_0", "agent_3"],
    "agent_2": ["agent_0", "agent_3"],
    "agent_3": ["agent_1", "agent_2"]
  }
}
```

In this configuration, each agent samples one neighbor per round according 
to `neighborhood`, aggregates only the layers whose names start with `policy`, 
and keeps layers matching `critic` private. Because the random neighbor choice 
is driven by a seeded NumPy generator, using the same `exp_seed` makes the 
communication pattern reproducible across runs, provided that the rest of the 
setup is unchanged.

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
