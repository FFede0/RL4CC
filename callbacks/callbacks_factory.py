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


class CallbacksFactory:
  """
  Factory of `Callbacks`
  """
  all_callbacks = {}
  
  @classmethod
  def register(cls, name: str):
    """
    Register the given `callbacks` under the provided name
    """
    def inner_wrapper(wrapped_class):
      cls.all_callbacks[name] = wrapped_class
      return wrapped_class
    return inner_wrapper
  
  @classmethod
  def get_type(cls, name: str):
    """
    Get the callbacks type according to the given name
    """
    callbacks = cls.all_callbacks.get(name)
    if not callbacks:
        raise ValueError(name)
    return callbacks
