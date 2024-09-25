import pickle
from typing import List, Tuple

import networkx as nx
import numpy as np

from general.entities import Point
from config import DATA_PATH


def create_graph_from_last_population(population: List[Point]) -> nx.DiGraph:
    def check_and_update_node(graph, node_id, metadata):
        if graph.has_node(node_id):
            graph.nodes[node_id].update(metadata)
        else:
            graph.add_node(node_id, **metadata)

    def add_node(graph: nx.DiGraph, point: Point):
        check_and_update_node(graph, point.id, point.get_metadata())
        if point.parents is not None:
            for parent in point.parents:
                graph.add_edge(parent.id, point.id)
                add_node(graph, parent)

    # Create a NetworkX graph
    graph = nx.DiGraph()
    for point in population:
        add_node(graph, point)

    return graph


def create_graph_from_population_log(population_log: List[List[Point]]) -> nx.DiGraph:
    all_points = [p for population in population_log for p in population]

    graph = nx.DiGraph()
    for point in all_points:
        graph.add_node(point.id, final=False, **point.get_metadata())

    for point in all_points:
        if point.parents is not None:
            for parent in point.parents:
                graph.add_edge(parent.id, point.id)

    for point in population_log[-1]:
        graph.nodes[point.id]['final'] = True

    return graph


def population_to_graph_matrices(population: List[Point]) -> Tuple[np.array, np.array]:
    # returns adj matrix, attribute matrix
    ...


def graph_to_pickle(graph: nx.Graph, filename: str):
    assert filename.endswith('.pkl'), 'Wrong format! Only .pkl accepted'
    fpath = f'{DATA_PATH}/pickle/{filename}'
    with open(fpath, "wb") as f:
        pickle.dump(graph, f)


def pickle_to_graph(filename: str, folder: str = '') -> nx.Graph:
    fpath = f'{DATA_PATH}{folder}/pickle/{filename}'
    with open(fpath, "rb") as f:
        G = pickle.load(f)
    return G
