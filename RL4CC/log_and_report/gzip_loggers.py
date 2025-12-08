"""
Copyright 2025 Federica Filippini

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
from ray.tune.utils.util import SafeFallbackEncoder
from ray.air.constants import EXPR_PROGRESS_FILE
from ray.tune.logger import Logger, CSVLogger
from ray.tune.utils import flatten_dict

from pathlib import Path
from typing import Dict
import gzip
import json
import csv
import io


class JsonGzipLogger(Logger):
  """
  A logger that writes results to a gzip-compressed JSON file.
  I use this logger instead of the standard one because the JSON data can be
  quite large, and compressing it helps reduce the file size.
  By default, in Ray RLlib's old stack API, a UnifiedLogger is used with the
  default configuration to log results in multiple formats, including plain
  JSON, CSV, TensorBoardX files, and a few additional minor files. The
  UnifiedLogger is essentially a combination of several specialized loggers,
  such as CsvLogger and JsonLogger.
  In addition, the Algorithm class extends the Trainable class, and the latter
  is responsible for creating the default logger when no specific options are
  provided. This is why the logger interface, along with classes like
  UnifiedLogger and JsonLogger, are located under the ray.train module.

  Args:
    config (dict): Configuration dictionary for the logger.
    logdir (str or Path): Directory where the log file will be saved.

  Attributes:
    file (gzip.GzipFile): The gzip-compressed file object used for logging.

  References:
    https://github.com/ray-project/ray/blob/master/python/ray/tune/logger/json.py#L30
    https://github.com/ray-project/ray/blob/master/python/ray/tune/logger/logger.py#L35
    https://github.com/ray-project/ray/blob/master/python/ray/tune/logger/unified.py#L19
    https://github.com/ray-project/ray/blob/master/python/ray/tune/trainable/trainable.py#L666
    https://github.com/ray-project/ray/blob/master/rllib/algorithms/algorithm.py#L207
  """

  def __init__(self, config, logdir):
    super().__init__(config, logdir)
    log_path = Path(logdir) / "result.json.gz"
    self.file = gzip.open(log_path, "wt", encoding="utf-8")

  def on_result(self, result):
    # The SafeFallbackEncoder class comes from Ray and has additional
    # handling for objects that cannot be encoded with the default encoder.
    json.dump(result, self.file, cls=SafeFallbackEncoder)
    self.file.write("\n")
    # We noted that Ray RLlib does not call close() when an experiment
    # terminates, so we need to continuously flush data to disk with every
    # write.
    self.flush()

  def update_config(self, config):
    # This method is unused but required by the parent class.
    pass

  def close(self):
    self.file.close()

  def flush(self):
    self.file.flush()


class CsvGzipLogger(CSVLogger):
  """Logs results to progress.csv under the trial directory.
  Automatically flattens nested dicts in the result dict before writing
  to csv:
    {"a": {"b": 1, "c": 2}} -> {"a/b": 1, "a/c": 2}
  """
  def __init__(self, config, logdir):
    super().__init__(config, logdir)

  def _maybe_init(self):
    """CSV outputted with Headers as first set of results."""
    if not self._initialized:
      progress_file = Path(self.logdir, f"{EXPR_PROGRESS_FILE}.gz")
      self._continuing = (
        progress_file.exists() and progress_file.stat().st_size > 0
      )
      self._file = gzip.open(progress_file, "wb")
      self._wrapper = io.TextIOWrapper(self._file, encoding='utf-8')
      self._csv_out = None
      # eval
      evaluation_file = Path(self.logdir, "evaluations.csv.gz")
      self._eval_continuing = (
        evaluation_file.exists() and evaluation_file.stat().st_size > 0
      )
      self._eval_file = gzip.open(evaluation_file, "wb")
      self._eval_wrapper = io.TextIOWrapper(self._eval_file, encoding='utf-8')
      self._eval_csv_out = None
      self._initialized = True

  def on_result(self, result: Dict):
      self._maybe_init()
      tmp = result.copy()
      if "config" in tmp:
        del tmp["config"]
      # print 'regular' results
      result = flatten_dict(tmp, delimiter="/")
      if self._csv_out is None:
        self._csv_out = csv.DictWriter(self._wrapper, result.keys())
        if not self._continuing:
          self._csv_out.writeheader()
      self._csv_out.writerow(
        {k: v for k, v in result.items() if k in self._csv_out.fieldnames}
      )
      self._file.flush()
      # print 'evaluation' result
      if "evaluation" in tmp:
        tmp["evaluation"]["after_training_iteration"] = tmp[
          "training_iteration"
        ]
        eval_result = flatten_dict(tmp["evaluation"], delimiter="/")
        if self._eval_csv_out is None:
          self._eval_csv_out = csv.DictWriter(
            self._eval_wrapper, eval_result.keys()
          )
          if not self._continuing:
            self._eval_csv_out.writeheader()
        self._eval_csv_out.writerow({
            k: v for k, v in eval_result.items() 
              if k in self._eval_csv_out.fieldnames  
        })
        self._eval_file.flush()

  def flush(self):
      super().flush()
      if self._initialized and not self._eval_file.closed:
        self._eval_file.flush()

  def close(self):
      super().close()
      if self._initialized:
        self._eval_file.close()
