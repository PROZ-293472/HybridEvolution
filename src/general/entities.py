from typing import List, Optional, Callable

import numpy as np


class Point:
    GLOBAL_ID = 0

    def __init__(self, coordinates: np.array, target_fun: Callable, parents: List = None):
        self.id = Point.GLOBAL_ID
        Point.GLOBAL_ID = Point.GLOBAL_ID + 1

        self.coordinates = coordinates
        self.parents = parents
        self.target_fun = target_fun

    def __repr__(self):
        return str(self.__dict__)

    @property
    def fitness(self):
        return self.target_fun(self.coordinates)

    def copy_basic_attributes(self):
        return Point(coordinates=self.coordinates, target_fun=self.target_fun)

    def get_metadata(self):
        return {'coordinates': self.coordinates, 'fitness': self.fitness}
