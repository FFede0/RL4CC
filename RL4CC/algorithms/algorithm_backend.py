"""
Copyright 2026 Federica Filippini

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
from abc import ABC, abstractmethod


class AlgorithmBackend(ABC):
  """
  Backend interface for algorithms supported by RL4CC.

  The backend hides the differences between RLlib algorithms and
  external algorithm implementations.
  """
  @abstractmethod
  def build(self):
    """
    Build the underlying algorithm, if required.
    """
    pass

  @abstractmethod
  def train(self) -> dict:
    """
    Perform one RL4CC training iteration.
    """
    pass

  @abstractmethod
  def evaluate(self) -> dict:
    """
    Perform one RL4CC evaluation step.
    """
    pass

  @abstractmethod
  def stop(self):
    """
    Release resources used by the algorithm.
    """
    pass

  @abstractmethod
  def last_iteration(self) -> int:
    """
    Return the current RL4CC training iteration.
    """
    pass

  @abstractmethod
  def save_checkpoint(
      self, manual: bool = False, path: str = None
    ) -> str:
    """
    Save an algorithm checkpoint.
    """
    pass

  @abstractmethod
  def compute_single_action(
      self, obs, explore: bool = False, weight = None
    ):
    """
    Compute one action.

    `weight` is ignored by single-objective/RLlib algorithms and is
    used by preference-conditioned MORL algorithms.
    """
    pass
