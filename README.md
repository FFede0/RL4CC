# RL4CC

The **R**einforcement **L**earning for the **C**omputing **C**ontinuum module 
provides a common interface to define environments and RL algorithms based 
on the [Ray RLLib](https://docs.ray.io/en/latest/rllib/index.html) library.

## How to add new RL methods

To expand the module with generators for new algorithms:
1. implement a suitable subclass of the base 
[`AlgoConfigGenerator`](algorithms/generators/algo_config_generator.py) 
(see, as an example, what is provided for the 
[PPO algorithm](algorithms/generators/ppo_config_generator.py))
2. add the new generator to the 
[generators factory](algorithms/generators_factory.py)
