import numpy as np
from src.transformer.transformer import TransformerModel

import pandas as pd
from typing import List, Tuple

from src.general.entities import Point
from src.evolution.algorithms import Evolution


class HybridModel:
    def __init__(self, ev: Evolution, transformer: TransformerModel, window_len: int, sequence_len: int, variant: int = 2):
        self.ev = ev
        self.model = transformer
        self.window_len = window_len
        self.sequence_len = sequence_len
        self.variant = variant
 


    @staticmethod
    def vectors_to_token(current_vector, prev_vector):
        cosine_sim = np.dot(current_vector, prev_vector) / (np.linalg.norm(current_vector) * np.linalg.norm(prev_vector))
        return (cosine_sim, np.linalg.norm(current_vector))


    @staticmethod
    def points_to_vectors(points: List[np.ndarray]) -> List[np.ndarray]:
        vectors = []
        for i, point in enumerate(points):
            if i == 0:
                continue
            vectors.append(point - points[i - 1])
        return vectors

    @staticmethod
    def population_log_to_token_sequence(population_log: pd.DataFrame) -> List[Tuple]:
        means = Evolution.get_populations_means(population_log)
        vectors = HybridModel.points_to_vectors(means)
        sequence = []
        for i, vector in enumerate(vectors):
            if i == 0:
                continue
            sequence.append(HybridModel.vectors_to_token(vector, vectors[i - 1]))
        return sequence

    @staticmethod
    def generate_vector(v, norm_ratio, cosine_similarity):
        norm_v = np.linalg.norm(v)
        v_normalized = v / norm_v  

        norm_u = norm_ratio * norm_v
        u_parallel = cosine_similarity * norm_u * v_normalized

        # Find an orthogonal vector to v
        if np.all(v == 0):
            raise ValueError("The input vector v is zero vector, can't define direction.")

        # Generate a random vector and make it orthogonal to v
        random_vector = np.random.randn(*v.shape)
        orthogonal_vector = random_vector - np.dot(random_vector, v_normalized) * v_normalized
        orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector)  # Normalize it

        u_perp = np.sqrt(1 - cosine_similarity ** 2) * norm_u * orthogonal_vector

        u = u_parallel + u_perp
        return u
    
    @staticmethod
    def get_point(sp, v):
        position = sp + v
        # fitness = obj_fun(position)
        return position
    
    @staticmethod
    def get_sequence_of_points(sp, vectors):
        seq = [sp]
        for vector in vectors:
            next_point = HybridModel.get_point(sp, vector)
            seq.append(next_point)

            sp = next_point
        return seq
    
    @staticmethod
    def get_sequence_of_vectors(sv, embeddings):
        seq = [sv]
        for embedding in embeddings:
            cosine_sim, u_norm = embedding[0], embedding[1]
            next_vector = HybridModel.generate_vector(sv, u_norm, cosine_sim)
            seq.append(next_vector)

            sp = next_vector
        return seq
    
    @staticmethod
    def get_sequence_of_fitness(points, obj_fun):
        return [obj_fun(point.astype(float)) for point in points]

    def generate_population_variant_1(self,input_vectors, output_sequence):
        output_vectors = HybridModel.get_sequence_of_vectors(sv=input_vectors[-1], embeddings=output_sequence)
        cumulative_vector = np.array(output_vectors).sum(axis=0)

        translated_pop = [
            Point(
                coordinates=point.coordinates + cumulative_vector,
                target_fun=point.target_fun)
            for point in self.ev.population
        ]

        return translated_pop

    def generate_population_variant_2(self, input_vectors, output_sequence):
        translated_pop = []
        for point in self.ev.population:
            output_vectors = self.get_sequence_of_vectors(sv=input_vectors[-1], embeddings=output_sequence)
            cumulative_vector = np.array(output_vectors).sum(axis=0)

            translated_pop.append(
                Point(
                    coordinates=point.coordinates + cumulative_vector,
                    target_fun=point.target_fun
                ))
        return translated_pop  


    def run(self, total_iters):
        """
        transformer wyznacza punkt środkowy populacji a następnie przesuwamy populację o wektor wyznaczony przez transformer
        """
        total_fitnesses = []
        for _ in range(int(total_iters / (self.window_len))):
            self.ev.run(self.window_len) 
            population_log = self.ev.population_log_to_df()
            input_log = population_log.iloc[:self.window_len * len(self.ev.population)]
            input_points = Evolution.get_populations_means(input_log)

            total_fitnesses += self.get_sequence_of_fitness(input_points, self.ev.population[0].target_fun)

            input_vectors = self.points_to_vectors(input_points)
            input_sequence = self.population_log_to_token_sequence(input_log)
            
            output_sequence = self.model.continue_sequence(
                input_sequence=input_sequence,
                continuation_len=self.sequence_len
            )[-self.sequence_len:]
            # print(output_sequence)

            if self.variant == 1:
                translated_population = self.generate_population_variant_1(input_vectors, output_sequence)

            else:
                translated_population = self.generate_population_variant_2(input_vectors, output_sequence)
            
            # self.ev.population = translated_population
            self.ev = Evolution(population=translated_population, mutate_fun=self.ev._mutate_fun, elitist=self.ev.elitist)

        return total_fitnesses



