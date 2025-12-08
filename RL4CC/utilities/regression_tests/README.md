## RL4CC Regression Tests

Run to check if any change modifed stable parts of the RL4CC library.

The existing regression tests concern algorithm configuration and training 
for PPO, DQN and SAC. Expected outputs are saved in suitable subdirectories 
accoding to the tested version of Ray (2.8.1, 2.10.0, 2.20.0). 

> [!CAUTION]
> When using python >= 3.10 and Ray >= 2.8.1, install numpy < 2.0.0 to avoid 
> issues with `np.product`.

Regression tests are automatically performed by running:

```
python run_regression_tests.py
```

The output of any test raising errors is saved in `regression_tests/ERRORS`, 
while regular outputs are saved in `regression_tests/LOGDIR`.

> [!WARNING]
> The evaluation interval in training experiments has been set to 1 to avoid 
> issues with a `num_env_steps_sampled_for_evaluation_this_iter` key appearing 
> in `info` when restoring a checkpoint (at least, it is saved in the 
> progress file right from the start). This has to be fixed as soon as 
> possible!
