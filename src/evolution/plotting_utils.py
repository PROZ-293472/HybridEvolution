from typing import List

from matplotlib import pyplot as plt

from general.entities import Point


def plot_population(
        population: List[Point],
        label=None,
        fig=None,
        ax=None
):
    try:
        assert len(population[0].coordinates) == 2
    except AssertionError:
        raise NotImplementedError('Plotting implemented only for 2D space')

    if fig is None:
        # Create a new figure and axis
        fig, ax = plt.subplots()
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Scatter Plot of 2D Coordinates')

    xs, ys = zip(*[p.coordinates for p in population])
    ax.scatter(xs, ys)
    plt.show()

    return fig, ax