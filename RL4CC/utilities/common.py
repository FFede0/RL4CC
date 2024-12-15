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
from typing import Tuple
import pandas as pd
import numpy as np
import json
import os


def not_defined(param: str, params_dict: dict) -> bool:
  """
  Return True if the given parameter is not set in a parameters dictionary
  """
  return not defined(param, params_dict)

def defined(param: str, params_dict: dict) -> bool:
  """
  Return True if the given parameter is set in a parameters dictionary.
  """
  return param in params_dict and params_dict[param] is not None

def load_config_file(filename: str) -> dict:
  """
  Load the configuration file whose name is provided as parameter
  (if available)
  """
  config = None
  if filename is not None and os.path.exists(filename):
    with open(filename, "r") as istream:
      config = json.load(istream)
  return config

def write_config_file(jconfig: str, dirname: str, filename: str) -> str:
  """
  Write the given configuration dictionary (in json format) into the
  provided directory with the given file name
  """
  os.makedirs(dirname, exist_ok = True)
  filename = os.path.join(dirname, filename)
  with open(filename, "w") as ostream:
    ostream.write(jconfig)
  return filename

def compare_dictionaries(d1: dict, d2: dict) -> Tuple[bool, list]:
  """
  Return True if the content of two dictionaries (possibly with nested keys)
  is equal
  """
  equal = True
  different_keys = []
  for key, val in d1.items():
    if key in d2:
      if isinstance(val, dict):
        e, d = compare_dictionaries(d1[key], d2[key])
        if not e:
          different_keys += d
          equal = False
      elif isinstance(val, list) or isinstance(val, tuple):
        if any([v1 != v2 for v1, v2 in zip(val, d2[key])]):
          different_keys.append(key)
          equal = False
      else:
        if key == "logdir":
          v1 = "/".join(val.split("/")[:-1])
          v2 = "/".join(d2[key].split("/")[:-1])
          if v1 != v2:
            different_keys.append(key)
            equal = False
        else:
          if val != d2[key]:
            different_keys.append(key)
            equal = False
    else:
      different_keys.append(key)
      return False, different_keys
  return equal, different_keys

def compute_deviation(
    baseline: pd.Series, target: pd.Series
  ) -> Tuple[pd.Series, float, float, float]:
  """
  Compute the deviation (and its minimum, maximum and average value) between
  a baseline and a target (increases as target becomes lower than baseline)
  """
  deviation = (baseline - target) / baseline
  m = float(deviation.min())
  M = float(deviation.max())
  avg = float(deviation.mean())
  return deviation, m, M, avg

# Thanks to: https://stackoverflow.com/a/47626762
class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    elif isinstance(obj, np.number):
      return obj.item()

    # Use the default JSON encoder for other types.
    return json.JSONEncoder.default(self, obj)

def update_json_file(filename: str, key: str, value):
  """
  Update the information written in the experiment progress file
  """
  json_obj = {}
  # load existing content (if any)
  if os.path.exists(filename):
    with open(filename, "r") as istream:
      json_obj = json.load(istream)
  # update
  json_obj[key] = value
  # write updated file
  with open(filename, "w") as ostream:
    ostream.write(json.dumps(json_obj, indent = 2))
