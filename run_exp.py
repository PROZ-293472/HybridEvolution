import os
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import benchmark_functions as bf

from src.general.entities import Point
from src.evolution.algorithms import Evolution
from src.hybrid.hybrid_model import HybridModel
from src.transformer.transformer import TransformerModel



def run_only_evolution(ev, iter_num):
    ev.run(iter_num)
    pop_log = ev.population_log_to_df()
    pop_means = Evolution.get_populations_means(pop_log)
    means_fitnesses = HybridModel.get_sequence_of_fitness(pop_means, ev.population[0].target_fun)

    return means_fitnesses


def run_experiment(
        model,
        target,
        dim,
        mutation_std_dev,
        pop_size,
        total_iterations,
        window_len,
        continuation_len,
        output_dir
        ):

    base_fname = f'{target.__name__}_{dim}_{int(mutation_std_dev)}_{pop_size}_{window_len}_{continuation_len}'

    def mutation_fun(coords: np.array, population_coords: np.array):
        mean = 0.0
        std_dev = mutation_std_dev
        gaussian_noise = np.random.normal(mean, std_dev, size=coords.shape)
        return coords + 1.0 * gaussian_noise

    population_mean = np.random.rand(dim) * 1000
    scale = 10.0
    target_fun = target(
        n_dimensions=dim,
        # attributes=np.random.uniform(1, 10, dim)
    )

    start_pop = [
        Point(
            coordinates=np.random.normal(size=dim, loc=population_mean, scale=scale),
            target_fun=target_fun)
        for _ in range(pop_size)
    ]

    v0s = []
    v1s = []
    v2s = []
    for _ in tqdm(range(10)):
        ev = Evolution(
            population=start_pop,
            mutate_fun=mutation_fun,
            elitist=False
        )

        h1 = HybridModel(
            transformer=deepcopy(model),
            ev=deepcopy(ev),
            window_len=20,
            sequence_len=20,
            variant=1
        )

        h2 = HybridModel(
            transformer=deepcopy(model),
            ev=deepcopy(ev),
            window_len=20,
            sequence_len=20,
            variant=2
        )
        v0 = run_only_evolution(
            ev=deepcopy(ev),
            iter_num=total_iterations,
        )

        v1 = h1.run(total_iters=total_iterations)
        v2 = h2.run(total_iters=total_iterations)

        v0s.append(v0)
        v1s.append(v1)
        v2s.append(v2)

    iter_cutoff = 600

    all_v0s = np.array([v[:iter_cutoff] for v in v0s])
    all_v1s = np.array([v[:iter_cutoff] for v in v1s])
    all_v2s = np.array([v[:iter_cutoff] for v in v2s])

    threshold = 60

    totals = {
        'target': target.__name__,
        'dim': dim,
        'v0': np.concatenate([np.array(v[threshold:threshold+1]) for v in v0s]),
        'v1': np.concatenate([np.array(v[threshold:threshold+1]) for v in v1s]),
        'v2': np.concatenate([np.array(v[threshold:threshold+1]) for v in v2s])
    }


    df = pd.DataFrame(totals)
    df.to_csv(f'{output_dir}/data/means_{base_fname}.csv', index=False)

    # Calculate mean of all iterations for each variant
    mean_v0 = np.mean(all_v0s, axis=0)
    mean_v1 = np.mean(all_v1s, axis=0)
    mean_v2 = np.mean(all_v2s, axis=0)

    # Initialize lists to collect smallest values and stability points
    min_values = {'v0': [], 'v1': [], 'v2': []}
 
    # Process each variant
    for variant_data, key in zip([v0s, v1s, v2s], ['v0', 'v1', 'v2']):
        for series in variant_data:
            truncated_series = series
            min_values[key].append(np.min(truncated_series))

    # # Calculate the mean of the smallest values and mean stability points for each variant
    # mean_smallest_values = {k: np.mean(v) for k, v in min_values.items()}
    # mean_stability_points = {k: np.mean(v) for k, v in stability_points.items()}

    # results = {'mean_smallest': mean_smallest_values, 'mean_stability': mean_stability_points}
    # with open(f'{output_dir}/data/{base_fname}.json', 'w') as json_file:
    #     json.dump(results, json_file)



    plt.figure(figsize=(10, 6))
    # Plot each line with reduced alpha
    for i in range(len(v0s)):
        plt.plot(v0s[i][:iter_cutoff], linestyle='--', color='b', alpha=0.25)
        plt.plot(v1s[i][:iter_cutoff], linestyle='--', color='g', alpha=0.25)
        plt.plot(v2s[i][:iter_cutoff], linestyle='--', color='r', alpha=0.25)

    # Plot mean lines with full opacity and labels
    plt.plot(mean_v0, linestyle='-', color='b', label='Pure Evolution', linewidth=2)
    plt.plot(mean_v1, linestyle='-', color='g', label='Variant 1', linewidth=2)
    plt.plot(mean_v2, linestyle='-', color='r', label='Variant 2', linewidth=2)

    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    plt.title(f'Uśredniona wartość punktu środkowego populacji dla funkcji {target.__name__}')

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.yscale('log')
    plt.savefig(f'{output_dir}/figs/{base_fname}_log.png')

    plt.yscale('linear')
    plt.savefig(f'{output_dir}/figs/{base_fname}_linear.png')






if __name__ == '__main__':
    model_name = 'model_40c52fd9-0797-480b-b884-6d74da285816_egg_50.pth'
    model_dir = 'models/' + model_name

    RESULTS_DIR = f'new_results/{model_name}'

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR+'/data')
        os.makedirs(RESULTS_DIR + '/figs')


    MODEL = torch.load(model_dir)
    DIM = 10
    MUTATION_STD_DEV = 10.0
    POP_SIZE = 20
    # TARGET = EggFunction
    WINDOW_LEN = 20
    CONTINUATION_LEN = 20
    TOTAL_ITERATIONS = 500

    targets = [
        # EggFunction,
        bf.Hypersphere,
        bf.Rastrigin,
        bf.Rosenbrock,
        bf.Ackley
    ]

    for TARGET in targets:
        print(TARGET)
        run_experiment(
            model=MODEL,
            target=TARGET,
            dim=DIM,
            mutation_std_dev=MUTATION_STD_DEV,
            pop_size=POP_SIZE,
            total_iterations=TOTAL_ITERATIONS,
            window_len=WINDOW_LEN,
            continuation_len=CONTINUATION_LEN,
            output_dir=RESULTS_DIR
        )