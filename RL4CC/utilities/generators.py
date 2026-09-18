from RL4CC.log_and_report.rl4cc_logger import Logger

from typing import Tuple


def _compute_num_steps_per_episode(env_config: dict) -> int:
    """
    Compute the number of steps per episode based on the environment 
    configuration
    """
    n_steps = None
    if all(k in env_config for k in ["min_time", "max_time", "time_step"]):
      min_time = env_config["min_time"]
      max_time = env_config["max_time"]
      time_step = env_config["time_step"]
      n_steps = (max_time - min_time) // time_step
    else:
      raise ValueError(
        "ERROR: not enough parameters to support `episodes` duration. "
        "Check if env_config.json includes `min_time`, `max_time`, "
        "`time_step`"
      )
    return n_steps


def _validate_key_usage(
    protected_keys: list,
    suggested_keys: list, 
    all_params: dict,
    logger: Logger
  ) -> Tuple[bool, bool]:
  """
  Checks if the user is setting any protected/suggested key and throws
  appropriate errors/warnings
  """
  # check if the user is setting any suggested key
  using_suggested_keys = any(
    k in all_params for k,_ , _ in suggested_keys
  )
  # check if the user is setting any protected key
  using_protected_keys = False
  for pk,_ in protected_keys:
    if pk in all_params:
      # prevent the user from improperly setting the environment config
      if pk == "env" or pk == "env_config":
        raise KeyError(
          "ERROR: `env` and `env_config` cannot be manually configured"
        )
      # prevent the user from improperly setting the evaluation interval
      elif pk == "evaluation_interval":
        raise KeyError(
          "ERROR: set the evaluation interval from `exp_config.json`"
        )
      # prevent the user from manually setting the logging directory
      elif pk == "logger_config":
        if "logdir" in all_params[pk]:
          raise KeyError(
            "ERROR: set a general logging directory from `exp_config.json`"
          )
      else:
        using_protected_keys = True
        # prevent the user from simultaneously setting protected and
        # suggested keys
        if using_suggested_keys:
          msg = "ERROR: mixing protected and suggested keys is forbidden"
          raise KeyError(
            msg + f" (protected key: `{pk}`)"
          )
        # raise a warning otherwise
        else:
          pv = all_params[pk]
          logger.warn(
            f"manually setting protected key `{pk}` with value: {pv}"
          )
  return using_suggested_keys, using_protected_keys