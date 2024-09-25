from copy import copy
from typing import List, Callable

import benchmark_functions
import pandas as pd
import numpy as np

from src.general.entities import Point


class Evolution:

    def __init__(
            self,
            population: List[Point],
            mutate_fun: Callable,
            elitist: bool = True,
            store_population_log: bool = True
    ):
        self.population = population
        self._mutate_fun = mutate_fun
        self._crossover_fun = None
        self.elitist = elitist
        self.population_log = [] if store_population_log else None

    @staticmethod
    # @nb.njit
    def _tournament_selection(population_fitness: np.ndarray, tournament_size: int) -> np.ndarray:
        population_size = population_fitness.shape[0]

        # rand_arrays = [np.random.choice(population_size, size=tournament_size, replace=False) for _ in range(population_size)]
        # ids_matrix = np.array(rand_arrays)

        ids_matrix = np.random.choice(population_size, size=(population_size, tournament_size))

        fitness_matrix = population_fitness[ids_matrix]

        min_fitness_ids = np.argmin(fitness_matrix, axis=1)
        best_ids = ids_matrix[np.arange(len(min_fitness_ids)), min_fitness_ids]

        return best_ids

    @staticmethod
    def _threshold_selection(population_fitness: np.ndarray) -> np.ndarray:
        ...

    def _selection(self, points: List[Point], mode: str) -> List[Point]:
        # points = sorted(points, key=lambda x: x.fitness)

        if mode == 'tournament':
            points_fitness = np.array([p.fitness for p in points])
            idx = self._tournament_selection(points_fitness, tournament_size=3)

        elif mode == 'threshold':
            raise NotImplementedError()

        return [points[i] for i in idx]


    def _new_generation(self, points: List[Point]) -> List[Point]:
        new_points = []
        if self._crossover_fun is not None:
            raise NotImplementedError()
        else:
            for point in points:
                successor = point.copy_basic_attributes()
                if successor.parents is None:
                    successor.parents = [point]
                else:
                    successor.parents.append(point)
                new_points.append(successor)
        return new_points

    def _mutation(self, points: List[Point]) -> List[Point]:
        new_points = copy(points)
        population_coords = np.array([point.coordinates for point in self.population])
        for point in new_points:
            point.coordinates = self._mutate_fun(point.coordinates, population_coords)
        return new_points

    def _succession(
            self,
            new_population: List[Point],
            old_population: List[Point],
            elite: bool = False) -> List[Point]:
        assert len(new_population) == len(old_population)
        if not elite:
            return new_population
        total_pop = new_population + old_population
        total_pop = sorted(total_pop, key=lambda x: x.fitness)
        return total_pop[:len(old_population)]

    @staticmethod
    def get_populations_means(population_log: pd.DataFrame) -> List[np.ndarray]:
        # population_log = deepcopy(population_log)
        # population_log['coordinates'] = population_log['coordinates'].apply(lambda x: str_to_np_array(x))
        means = population_log.groupby(by='population_num')['coordinates'].mean().to_list()
        population_log
        return means
    
    def iterate(self):
        pop = self._selection(self.population, mode='tournament')
        pop = self._new_generation(pop)
        pop = self._mutation(pop)
        new_pop = self._succession(pop, self.population, elite=self.elitist)

        self.population = new_pop
        if self.population_log is not None:
            self.population_log.append(self.population)

    def run(self, N: int):
        if self.population_log is not None:
            self.population_log.append(self.population)

        for _ in range(N):
            self.iterate()

    def get_best_point(self) -> Point:
        return sorted(self.population, key=lambda x: x.fitness)[0]

    def population_to_df(self, population=None):
        if not population:
            population = self.population

        list_of_points = []
        for point in population:
            d = point.get_metadata()
            d['parents_ids'] = [parent.id for parent in point.parents] if point.parents else None
            d['point_id'] = point.id
            list_of_points.append(d)
        return pd.DataFrame(list_of_points)

    def population_log_to_df(self):
        dfs = []
        for i, population in enumerate(self.population_log):
            df = self.population_to_df(population)
            df['population_num'] = i
            dfs.append(df)
        return pd.concat(dfs, axis=0)


if __name__ == '__main__':
    dim = 100
    target = benchmark_functions.Hypersphere
    pop_size = 10000
    mutation_std_dev = 0.5
    mutation_rate = 1.0

    target_fun = target(n_dimensions=dim)
    start_population = [
        Point(coordinates=np.random.random(size=dim) * 10, target_fun=target_fun)
        for _ in range(pop_size)
    ]


    def mutation_fun(coords: np.array, population_coords: np.array):
        mean = 0.0
        # std_dev = np.std(population_coords)
        std_dev = mutation_std_dev
        gaussian_noise = np.random.normal(mean, std_dev, size=coords.shape)
        return coords + mutation_rate * gaussian_noise

    ev = Evolution(
        population=start_population,
        mutate_fun=mutation_fun
    )

    ev.run(3)