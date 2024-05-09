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
`"callbacks/base_callbacks/BaseCallbacks"`).

Sample `ray_config.json` files for [PPO](ray_config_ppo.json.template) and 
[DQN](ray_config_dqn.json.template) are provided.

### How to use custom Policy models

Ray supports the use of Torch and TF models using the ModelV2 implementation.

One can easily implement their own custom network by extending either the [Torch](../models/base_torch_model.py) or the
[Tensorflow](../models/base_tf_model.py) Base models, and then registering in Ray's Models catalog as can be seen in the
[models initialization file](../models/__init__.py), where 2 custom models we built are registered and can be used off the shelf.

To actually indicate the use of a custom model after building one's own model by extending the base models
or using the custom models provided, one must specify the **name** of the model as was registered in Ray's models
catalog, and providing the accompanying custom model config dictionary in the Ray config dictionary. 

For example, using the provided [CustomTorchModel](../models/custom_torch_model.py):

  - 1- Make sure the model is registered (in this case it is already registered [here](../models/__init__.py) under
  the name **"custom_torch_model"**)

  - 2- In the ray_config json file, under **"training"** dictionary there is a **"model"**
dictionary, inside this dictionary the custom  model will be defined which will overwrite the
generic RLModule model.

 - 3- The custom model's name as registered in the Ray's model catalog must be passed using the
**"custom_model"** key, while its respective config must be passed under the **"custom_model_config"
key.

This would look like this inside the ray config file:

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
        },
      },
    .
    .
  }
 .
 .
```

:warning::warning::warning: The framework of the model **must** match the framework passed in the ray config
file, otherwise this will result in runtime errors.

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
- `tune_config_file`: path to the `tune_config.json` file described [below](#tuner-configuration-file)
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

## Configure hyperparameter tuning

### `Tuner` configuration file

In order to perform a hyperparameter tuning experiment, a `tune_config_path` must
be indicated in the exp_config.json file.

The `tune_config.json` file includes **three** mandatory parameter, which 
are related to the number of tune trials and their evaluation in the tuning experiment execution. 

These are:
- `num_tune_trials`: The number of tuning trials to be run in parallel. These trials will sample from the Tune search 
spaces indicated in the ray_config_*algorithm.json file, the examples given at the next section [below](#configuring-the-search-space-for-parameters) should help
explain this better.
- `metric`: The metric used to evaluate the performance of a given set of parameters in a trial.
- `mode`: The mode on which the values returned by the metric are evaluated, for instance; setting
this parameter to **"max"** when considering an **"epsiode_reward_mean"** metric, will place the trial 
that returns the highest numerical value of the mean reward as the best trials.


**Optional parameters**:
- `search_algorithm`: A search algorithm as specified in tune.search page [here](https://docs.ray.io/en/latest/tune/api/suggestion.html#hyperopt-tune-search-hyperopt-hyperoptsearch)
- `scheduler`: A scheduler as specified in tune.schedulers page [here](https://docs.ray.io/en/latest/tune/api/schedulers.html#tune-scheduler-hyperband)

**Note**: currently, only the **HyperOpt** search algorithm and the **ASHAScheduler** are implemented.

Sample configuration:

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
  }
}
```
### Configuring the `search space` for parameters

The Tuner configurations file `tune_config.json` prescribed in the section [above](#tuner-configuration-file)
is responsible for the behaviour of the **Tuner** and how it handles the **Trials**  running in parallel. However,
to actually fine-tune algorithm parameters, the user should indicate this in the algorithm's respecive
`ray_config.json` file.

This can be simply done by providing the **search_space** requirements for each parameter as a tune.**search_space string. 
For example, if the learning rate for the PPO algorithm is to be tuned, locate the required 
parameter in the ray_config_ppo.json file and set it as:
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

In the above example, the **Tuner** will sample **num_tune_trials** (present in the `tune_config.json` file) trials, each trial will run for the specified
number of **max_iterations** (present in the `exp_config.json` file), where each trial will have a different `lr` sampled from the `loguniform` distribution 
over the range of `(1e-4, 1e-1)`.

For more about tune's search spaces go [here](https://docs.ray.io/en/latest/tune/api/search_space.html).

**Note**: When the user tries to run a tuning experiment without specifying a `tune_config.json` or its
respective path in the `exp_config.json` file, the execution will be interrupted prompting the user to 
provide them.

