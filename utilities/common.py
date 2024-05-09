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
import json
import os


def not_defined(param: str, params_dict: dict) -> bool:
  """
  Return True if the given parameter is not set in a parameters dictionary
  """
  return (param not in params_dict) or (params_dict[param] is None)

def load_config_file(filename: str) -> dict:
  """
  Load the configuration file whose name is provided as parameter 
  (if available)
  """
  config = None
  if os.path.exists(filename):
    with open(filename, "r") as istream:
      config = json.load(istream)
  return config

def write_config_file(jconfig: str, dirname: str, filename: str):
  """
  Write the given configuration dictionary (in json format) into the 
  provided directory with the given file name
  """
  os.makedirs(dirname, exist_ok = True)
  with open(os.path.join(dirname, filename), "w") as ostream:
    ostream.write(jconfig)

def compare_dictionaries(d1: dict, d2: dict) -> bool:
  """
  Return True if the content of two dictionaries (possibly with nested keys) 
  is equal
  """
  equal = True
  for key, val in d1.items():
    if key in d2:
      if isinstance(val, dict):
        equal = equal and compare_dictionaries(d1[key], d2[key])
      elif isinstance(val, list) or isinstance(val, tuple):
        equal = equal and all([v1 == v2 for v1, v2 in zip(val, d2[key])])
      else:
        if key == "logdir":
          v1 = "/".join(val.split("/")[:-1])
          v2 = "/".join(d2[key].split("/")[:-1])
          equal = equal and (v1 == v2)
        else:
          equal = equal and (val == d2[key])
    else:
      return False
  return equal
