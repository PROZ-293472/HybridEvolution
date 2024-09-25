from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Minimum:
    score: float
    position: List[float]


class ObjectiveFunction(ABC):
    def __init__(self, n_dimensions: int):
        self.n_dimensions = n_dimensions

    @abstractmethod
    def __call__(self, *args, **kwargs) -> float:
        pass

    @abstractmethod
    def minima(self) -> List[Minimum]:
        pass

    @property
    @abstractmethod
    def _name(self) -> str:
        pass


class ZeroFunction(ObjectiveFunction):

    @property
    def _name(self) -> str:
        return 'zero'

    def __call__(self, *args, **kwargs):
        return 0.0

    def minima(self) -> List[Minimum]:
        minimum = Minimum(
            score=0.0,
            position=[0.0 for _ in range(self.n_dimensions)]
        )
        return [minimum]


class EggFunction(ObjectiveFunction):
    def __init__(self, n_dimensions: int, attributes: np.ndarray):
        super().__init__(n_dimensions)
        self.attributes = attributes

    @property
    def _name(self) -> str:
        return 'egg'

    def __call__(self, point: np.ndarray):
        return np.dot(point**2, self.attributes)

    def minima(self) -> List[Minimum]:
        minimum = Minimum(
            score=0.0,
            position=[0.0 for _ in range(self.n_dimensions)]
        )
        return [minimum]

