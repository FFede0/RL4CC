"""
Copyright 2024 Federica Filippini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from RL4CC.utilities.common import update_json_file

from ray.tune.progress_reporter import TuneReporterBase
from ray.tune.progress_reporter import _get_time_str, _get_trials_by_state
from ray.tune.utils import unflattened_lookup
from ray.tune.experiment.trial import Trial

from typing import Dict, List, Optional, Union
import time
try:
  from collections.abc import Mapping
except ImportError:
  from collections import Mapping


class BaseProgressReporter(TuneReporterBase):

  def __init__(
      self,
      *,
      metric_columns: Optional[Union[List[str], Dict[str, str]]] = None,
      parameter_columns: Optional[Union[List[str], Dict[str, str]]] = None,
      total_samples: Optional[int] = None,
      max_progress_rows: int = 20,
      max_error_rows: int = 20,
      max_column_length: int = 20,
      max_report_frequency: int = 5,
      infer_limit: int = 3,
      print_intermediate_tables: Optional[bool] = None,
      metric: Optional[str] = None,
      mode: Optional[str] = None,
      sort_by_metric: bool = False,
      progress_file: str = None
    ):
    super(BaseProgressReporter, self).__init__(
      metric_columns = metric_columns,
      parameter_columns = parameter_columns,
      total_samples = total_samples,
      max_progress_rows = max_progress_rows,
      max_error_rows = max_error_rows,
      max_column_length = max_column_length,
      max_report_frequency = max_report_frequency,
      infer_limit = infer_limit,
      print_intermediate_tables = print_intermediate_tables,
      metric = metric,
      mode = mode,
      sort_by_metric = sort_by_metric,
    )
    self.progress_file = progress_file

  def report(self, trials: List[Trial], done: bool, *sys_info: Dict):
    #super().report(trials, done, *sys_info)
    # get info on running time
    current_time_str, running_for_str = _get_time_str(self._start_time, time.time())
    update_json_file(self.progress_file, "current_time", current_time_str)
    update_json_file(self.progress_file, "tune_running_for", running_for_str)
    # system information (scheduling algorithm and resource usage)
    update_json_file(self.progress_file, "sys_info", sys_info)
    # get trials in different states
    num_trials = len(trials)
    trials_by_state = _get_trials_by_state(trials)
    num_trials_strs = [
        "{} {}".format(len(trials_by_state[state]), state)
        for state in sorted(trials_by_state)
    ]
    update_json_file(self.progress_file, "trials_state", num_trials_strs)
    # get information on the current best trial
    trial, metric = self._current_best_trial(trials)
    if trial is not None and self.progress_file is not None:
      # get metric value and trial parameters
      metric_val = unflattened_lookup(metric, trial.last_result, default=None)
      config = trial.last_result.get("config", {})
      parameter_columns = self._parameter_columns or list(config.keys())
      if isinstance(parameter_columns, Mapping):
        parameter_columns = parameter_columns.keys()
      params = {p: unflattened_lookup(p, config) for p in parameter_columns}
      # write information to the file
      update_json_file(self.progress_file, "best_trial_id", trial.trial_id)
      update_json_file(self.progress_file, "best_trial_metric", metric)
      update_json_file(self.progress_file, "best_trial_value", metric_val)
      # update_json_file(self.progress_file, "best_trial_config", params)

