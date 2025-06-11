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
from datetime import datetime
import sys


class Logger:
    def __init__(
        self, 
        name: str = "RL4CCLogger", 
        out_stream = sys.stdout, 
        err_stream = sys.stderr, 
        verbose: int = 0
      ):
      self.name = name
      self.out_stream = out_stream
      self.err_stream = err_stream
      self.verbose = verbose           

    def __getstate__(self):
      """
      Method to support pickling/unpickling of Logger objects
      """
      d = self.__dict__.copy()
      if "out_stream" in d:
        if type(d["out_stream"]) == "_io.TextIOWrapper":
          l = [d["out_stream"].name, d["out_stream"].mode]
        else:
          l = [d["out_stream"].name]
        d["out_stream"] = l
      if "err_stream" in d:
        if type(d["err_stream"]) == "_io.TextIOWrapper":
          l = [d["err_stream"].name, d["out_stream"].mode]
        else:
          l = [d["err_stream"].name]
        d["err_stream"] = l
      return d

    def __setstate__(self, d: dict):
      """
      Method to support pickling/unpickling of Logger objects
      """
      if "out_stream" in d:
        stream = Logger.parse_wrapper(d["out_stream"])
        d["out_stream"] = stream
      if "err_stream" in d:
        stream = Logger.parse_wrapper(d["err_stream"])
        d["err_stream"] = stream
      self.__dict__.update(d)
    
    def prepare_logging(self, msg: str, msg_type: str, msg_level: int) -> str:
      """
      Prepares a string for logging, adding the timestamp, the Logger name 
      and the message level and type
      """
      message = "{} [{}] (level {}) {}: {}".\
                  format(datetime.now(), self.name, msg_level, msg_type, msg)
      return message
    
    def log(self, message: str, v: int = 0):
      """
      Prints the given message if the given verbosity level is exceeded
      """
      if self.verbose >= v:
        full_message = self.prepare_logging(message, "INFO", v)
        print(full_message, file = self.out_stream, flush = True)
    
    def warn(self, message: str):
      """
      Prints a warning message
      """
      full_message = self.prepare_logging(message, "WARNING", 0)
      print(full_message, file = self.out_stream, flush = True)

    def err(self, message: str):
      """
      Prints an error message
      """
      full_message = self.prepare_logging(message, "ERROR", 0)
      print(full_message, file = self.err_stream, flush = True)
    
    def breakline(self, v: int = 0):
      """
      Prints a line of --- to interrupt the message
      """
      if self.verbose >= v:
        full_message = self.prepare_logging("-" * 80, "", 0)
        print(full_message, file = self.out_stream, flush = True)
    
    @staticmethod
    def parse_wrapper(wrapper: list):
      """
      Converts a list of stream properties into an actual stream
      The `wrapper` list is defined as [file_name, mode] if the stream is a 
      _io.TextIOWrapper, while it stores [stdout] if the stream is sys.stout 
      ([sterr], respectively)
      """
      if len(wrapper) > 1:
        filename = wrapper[0]
        mode = wrapper[1]
        stream = open(filename, mode)
      elif wrapper[0] == "stdout":
        stream = sys.stdout
      else:
        stream = sys.stderr
      return stream
