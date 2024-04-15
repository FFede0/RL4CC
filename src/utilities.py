import json
import os


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
