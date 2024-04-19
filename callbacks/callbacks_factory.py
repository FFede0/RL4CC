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
from callbacks.base_callbacks import BaseCallbacks


class CallbacksFactory:
  """
  Factory of `Callbacks`
  """
  def __init__(self):
    self.all_callbacks = {}
  
  def register(self, name: str, callbacks: BaseCallbacks):
    """
    Register the given `callbacks` under the provided name
    """
    self.all_callbacks[name] = callbacks
  
  def get_type(self, name: str):
    """
    Get the callbacks type according to the given name
    """
    callbacks = self.all_callbacks.get(name)
    if not callbacks:
        raise ValueError(name)
    return callbacks


## Factory initialization
CBfactory = CallbacksFactory()
CBfactory.register("BaseCallbacks", BaseCallbacks)
