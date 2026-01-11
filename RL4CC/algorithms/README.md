

> [!NOTE] The RL4CC library currently supports the Ray implementation of the PPO, SAC 
and DQN algorithms. Moreover, it implements the MAPPO algorithm, i.e., a PPO 
version that exploits a centralized critic model.

General details about the algorithms configuration are provided in the 
configurations [README](../config_files/README.md#ray-algorithm-configuration) 
file. Specific parameters related to the MAPPO implementation are instead 
reported below.

## MAPPO

The MAPPO algorithm is implemented based on Ray PPO algorithm (old API 
stack), extended with a custom implementation of a 
[centralized critic model](../models/centralized_critic_model.py). 
This model includes a fully-connected network that learns to estimate the 
central value function. It is intended for fully cooperative or mixed 
cooperative/competitive multi-agent environments with more than one opponent.

At a high level:

- Each agent’s **policy** only sees its own observation when acting.
- The **critic** sees the agent’s observation plus information about other 
agents (opponents), combined either by **concatenation** or by **averaging**.
- The module integrates cleanly with RLlib’s PPO implementation and model 
registry.

RL4CC currently exposes:

- `MAPPOConfig`: RLlib `PPOConfig` subclass with `MAPPO` as `algo_class`.
- `MAPPO`: RLlib `PPO` subclass that uses a custom centralized-critic PPO 
policy.
- `CCPPOTorchPolicy`: a Torch PPO policy with centralized-critic loss and 
postprocessing.
- `CustomTorchCCModel`: a Torch model implementing the centralized critic, 
configurable via `custom_model_config`.

RL4CC:

- Registers `MAPPO` in RLlib’s algorithm registry.
- Registers `CCPPOTorchPolicy` in RLlib’s policy registry.
- Registers `CustomTorchCCModel` under the name `"centralizedcritic"` 
in `ModelCatalog`.

### How the algorithm works

#### Centralized value, decentralized actions

The core idea is **centralized critic, decentralized execution**:

- **Actors**: each agent’s policy network receives only its 
**own observation** and outputs its action distribution (e.g., Dirichlet for 
continuous actions).
- **Critic**: a centralized value network gets:
  - The agent’s own observation.
  - Information about all other agents’ observations and actions.
  - It outputs a scalar **state-value estimate** used by PPO’s GAE and value 
  loss.

Concretely:

- `CCPPOTorchPolicy` extends `PPOTorchPolicy` and a `CentralizedValueMixin` 
that exposes `compute_central_vf`.
- In `postprocess_trajectory`, the policy:
  - Collects other agents’ **observations** and **actions** from 
  `other_agent_batches`.
  - Aggregates them according to a chosen **mode** (`"concat"` or `"avg"`, see 
  [below](#concatenation-vs-averaging-mode)).
  - Stores them into extra fields:
    - `self.OPPONENT_OBS` (string `"opponent_obs"`)
    - `self.OPPONENT_ACTION` (string `"opponent_action"`)
  - Overwrites `SampleBatch.VF_PREDS` with predictions from the 
  **centralized** value function.
- In `loss`, the policy:
  - Temporarily overrides `model.value_function()` so that PPO’s value loss 
  uses the **central** value computed from:
    - `CUR_OBS`
    - `OPPONENT_OBS`
    - `OPPONENT_ACTION`
  - Calls the base PPO loss, then restores the original value function.

#### Custom model: `CustomTorchCCModel`

`CustomTorchCCModel` subclasses RLlib’s example `TorchCentralizedCriticModel` 
and replaces its central value head:

- Uses `custom_model_config` to configure:
  - `n_agents`: total number of agents in the environment.
  - `central_vf_fcnet_hiddens`: size of the hidden layer(s) of the central 
  critic MLP.
  - `central_vf_fcnet_activation`: activation function of the hidden layer(s) 
  of the central critic MLP (the same for all layers).
  - `mode`: `"concat"` or `"avg"` (see next section).
- Computes the input size to the central critic based on `mode`:
  - For `"concat"`:
    - `n_obs = n_agents` (self obs + all opponents’ obs).
    - `n_acts = n_agents - 1` (opponents’ actions only).
  - For `"avg"`:
    - `n_obs = 2` (self obs + averaged opponents’ obs).
    - `n_acts = 1` (averaged opponents’ actions).
- Builds `self.central_vf` as:
  - `SlimFC` -> `SlimFC` MLP with possibly several layers (depending on the 
  elements in `central_vf_fcnet_hiddens`) and scalar output.
- In `central_value_function`, concatenates:
  - `obs` (self),
  - `opponent_obs` (aggregated opponents’ obs),
  - `opponent_actions` (aggregated opponents’ actions)

  into a single feature vector and feeds it to `self.central_vf`.

The policy and model must see consistent shapes; this is ensured by using the 
same `mode` and `n_agents` configuration in both places via 
`custom_model_config`.

Example of configuration:

```
"model": {
  "custom_model": "centralizedcritic",
  "custom_model_config": {
    "central_vf_fcnet_hiddens": [64, 64],
    "central_vf_fcnet_activation": "tanh",
    "mode": "concat", # or "avg"
  }
}
```

> [!CAUTION]
> The `n_agents` parameter **must not** be defined manually, it is 
> automatically filled according to the number of agents in the environment.

##### Concatenation vs averaging (`mode`)

The `mode` parameter in `custom_model_config` controls how information from 
multiple opponents is aggregated **before** it is passed to the centralized 
critic. 

The `"concat"` mode:
- Forms a large feature vector:
  - \([o^{(1)}, \dots, o^{(N)}, a^{(1)}, \dots, a^{(N)}]\), where each 
  opponent keeps its own slice.
- In code:
  - `opponent_obs = np.concatenate(op_obs_list, axis=1)` → shape 
  `[B, N * obs_dim]`
  - `opponent_actions = np.concatenate(op_act_list, axis=1)` → shape 
  `[B, N * act_dim]`.

**Advantages**

- The critic can learn **opponent-specific effects** (e.g., “opponent 3 being 
aggressive matters more than opponent 1”).  
- It can capture joint interactions like “if opponent A does X and opponent B 
does Y, the value is high”, since it sees all opponents individually and 
simultaneously.

**Disadvantages**

- Input dimension grows **linearly** with the number of agents:
  - More parameters,
  - More data required,
  - Higher risk of overfitting on small datasets.
- Requires a **fixed, meaningful ordering** of opponents; permuting agents 
changes the input, so the critic is **not permutation-invariant**.
- Harder to scale when the number of agents varies between episodes unless 
you introduce padding and masking.

**When to choose `"concat"`**

- Small, fixed number of agents (e.g., 2–4) and **identity/role matters**:
  - Agents occupy fixed positions,
  - You have designated roles (“leader”, “follower”).
- You care about modeling how specific other agents affect the value function, 
not just the overall crowd.

The `"avg"` mode:
- Reduces opponents to an aggregate:

  - \(\bar{o} = \frac{1}{N}\sum_i o^{(i)}\),  
    \(\bar{a} = \frac{1}{N}\sum_i a^{(i)}\).

  So the critic sees the **average** opponent state and behavior.
- In code:
  - `opponent_obs = np.mean(np.stack(op_obs_list, axis=0), axis=0)` → 
  `[B, obs_dim]`
  - `opponent_actions = np.mean(np.stack(op_act_list, axis=0), axis=0)` → 
  `[B, act_dim]`.

**Advantages**

- Input dimension is **independent of the number of agents**:
  - Much more scalable with large or variable N.
- Naturally **permutation-invariant**:
  - Swapping opponent identities does not change the aggregated features.
- Simpler critic:
  - Fewer parameters,
  - Often more stable and data-efficient when many agents are present.

**Disadvantages**

- Loses opponent **identity and heterogeneity**:
  - “One very aggressive + one very passive” can look similar to “two medium” 
  after averaging.
- Cannot easily represent fine-grained interactions between specific opponents.

**When to choose `"avg"`**

- Many agents with roughly symmetric roles, where only the 
**distribution or density** of opponents matters (e.g., crowding, mean-field 
settings).
- Number of opponents is large or changes over time and you want a 
**fixed-size** critic input.
- You prioritize simpler, more stable training rather than maximal 
expressiveness.

###### Practical guidance

- **Small fixed N (e.g., 2–4) and roles matter** → use **`"concat"`**, with a 
well-defined ordering of opponents (e.g., sorted by agent ID or position).  
- **Large or variable N, homogeneous roles** → use **`"avg"`**, so the critic 
is invariant to permutations and independent of agent count.

If `"avg"` turns out too coarse, a natural extension is to replace the 
hard-coded aggregation with an **attention** or **graph-based** centralized 
critic that learns its own permutation-invariant pooling instead of strict 
concatenation or mean.

> [!CAUTION]
> The centralized critic model implementation is available only for the 
> Torch module; Tensorflow is currently not supported.

## How to add new RL methods

To expand the module with generators for new algorithms:
1. implement a suitable subclass of the base 
[`AlgoConfigGenerator`](generators/algo_config_generator.py) 
(see, as an example, what is provided for the 
[PPO algorithm](generators/ppo_config_generator.py))
2. add the new generator to the 
[generators factory](generators_factory.py)
