#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

"""
Remove elements from a dataset using a genetic algorithm to try
and optimize for constant overlap between factors.
- Produces a boolean mask that can be applied to a dataset
"""

import logging
import multiprocessing
import os
import random
import warnings
from datetime import datetime
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scipy.stats import gmean

import research.util as H
from disent.dataset.wrapper import MaskedDataset
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.seeds import seed
from disent.util.strings.fmt import make_box_str
from research.e01_visual_overlap.util_compute_traversal_dists import cached_compute_all_factor_dist_matrices


log = logging.getLogger(__name__)


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def cxTwoPointNumpy(ind1: np.ndarray, ind2: np.ndarray):
    """
    Executes a two-point crossover on the input individuals. The two individuals
    are modified in place and both keep their original length.
    - Similar to tools.cxTwoPoint but modified for numpy arrays.
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1
    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
    return ind1, ind2


_NP_AGGREGATE_FNS = {
    'sum': np.sum,
    'mean': np.mean,
    'gmean': gmean,  # no negatives
    'max': lambda a, axis, dtype: np.amax(a, axis=axis),  # propagate NaNs
    'min': lambda a, axis, dtype: np.amin(a, axis=axis),  # propagate NaNs
    'std': np.std,
}


def np_aggregate(array, mode: str, axis=0, dtype=None):
    try:
        fn = _NP_AGGREGATE_FNS[mode]
    except KeyError:
        raise KeyError(f'invalid aggregate mode: {repr(mode)}, must be one of: {sorted(_NP_AGGREGATE_FNS.keys())}')
    result = fn(array, axis=axis, dtype=dtype)
    if dtype is not None:
        result = result.astype(dtype)
    return result


# ========================================================================= #
# BASE BOOLEAN GA                                                           #
# ========================================================================= #


class BooleanMaskGA(object):

    """
    Based on:
    - https://github.com/lmarti/evolutionary-computation-course/blob/master/AEC.06%20-%20Evolutionary%20Multi-Objective%20Optimization.ipynb
    """

    def __init__(
        self,
        toolbox_new_individual,
        toolbox_eval_individual,
        # objective
        objective_weights: Tuple[float, ...] = (1.0,),
        # population
        population_size: int = 256,
        num_generations: int = 100,
        # mutation
        mate_probability: float = 0.5,    # probability of mating two individuals
        mutate_probability: float = 0.2,  # probability of mutating an individual
        mutate_bit_flip_prob: float = 0.05,
        # job
        n_jobs: int = min(os.cpu_count(), 16),
    ):
        super().__init__()
        self.objective_weights    = tuple(objective_weights)
        self.population_size      = population_size
        self.num_generations      = num_generations
        self.mate_probability     = mate_probability
        self.mutate_probability   = mutate_probability
        self.mutate_bit_flip_prob = mutate_bit_flip_prob
        self.n_jobs               = n_jobs
        # toolbox functions
        self._toolbox_new_individual  = toolbox_new_individual
        self._toolbox_eval_individual = toolbox_eval_individual

    @property
    def parameters(self) -> Dict[str, Any]:
        return dict(
            objective_weights=self.objective_weights,
            # population
            population_size=self.population_size,
            num_generations=self.num_generations,
            # mutation
            mate_probability=self.mate_probability,
            mutate_probability=self.mutate_probability,
            mutate_bit_flip_prob=self.mutate_bit_flip_prob,
            # job
            n_jobs=self.n_jobs,
        )

    def _create_toolbox(self):
        creator.create("Fitness", base.Fitness, weights=self.objective_weights)
        creator.create("Individual", np.ndarray, fitness=creator.Fitness)
        # create toolbox!
        toolbox = base.Toolbox()
        # objects
        toolbox.register("individual", self._toolbox_new_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # evaluation
        toolbox.register("evaluate", self._toolbox_eval_individual)
        # mutation
        toolbox.register("mate",   cxTwoPointNumpy)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutate_bit_flip_prob)
        toolbox.register("select", tools.selNSGA2)
        # workers
        pool = None
        if self._num_workers > 1:
            pool = multiprocessing.Pool(processes=self._num_workers)
            toolbox.register('map', pool.map)
        # done!
        return toolbox, pool

    @property
    def _num_workers(self):
        if self.n_jobs > 0:   return self.n_jobs
        elif self.n_jobs < 0: return max(multiprocessing.cpu_count() + 1 + self.n_jobs, 1)
        else:                 raise ValueError('`n_jobs == 0` is invalid!')

    def fit(self) -> (List[np.ndarray], tools.Statistics, tools.HallOfFame):
        toolbox, pool = self._create_toolbox()
        # create new population
        population: List[np.ndarray] = toolbox.population(n=self.population_size)
        # allow individuals to be compared
        hall_of_fame = tools.HallOfFame(5, similar=np.array_equal)
        # create statistics tracker
        stats_fitness = tools.Statistics(lambda ind: ind.fitness.values)
        stats_fitness.register('min',  np.min,  axis=0)
        stats_fitness.register('max',  np.max,  axis=0)
        stats_fitness.register('mean', np.mean, axis=0)
        stats_fitness.register('std',  np.std,  axis=0)
        stats_mask    = tools.Statistics(lambda ind: ind.sum())
        stats_mask.register('min',  np.min,  axis=0)
        stats_mask.register('max',  np.max,  axis=0)
        stats_mask.register('mean', np.mean, axis=0)
        stats_mask.register('std',  np.std,  axis=0)
        stats_size    = tools.Statistics(lambda ind: ind.size)
        stats_size.register('max',  np.max,  axis=0)
        stats = tools.MultiStatistics(A_size=stats_size, B_mask=stats_mask, C_fitness=stats_fitness)
        # run genetic algorithm
        algorithms.eaMuPlusLambda(
            population=population,
            toolbox=toolbox,
            mu=self.population_size,
            lambda_=self.population_size,
            cxpb=self.mate_probability,
            mutpb=self.mutate_probability,
            ngen=self.num_generations,
            stats=stats,
            halloffame=hall_of_fame,
        )
        # cleanup pool
        if pool is not None:
            pool.close()
            pool.join()
        # done!
        return population, stats, hall_of_fame


# ========================================================================= #
# MASKING GA                                                                #
# ========================================================================= #


def _toolbox_new_individual(size: int) -> 'creator.Individual':
    # TODO: create arrays of shape, not size
    mask = np.random.random(size) < (0.8 * np.random.random() + 0.1)
    # mask = np.random.randint(0, 2, size=size, dtype='bool')
    return creator.Individual(mask)


def _toolbox_eval_individual(
    individual: np.ndarray,
    gt_dist_matrices: np.ndarray,
    factor_sizes: Tuple[int, ...],
    fitness_mode: str,
    obj_mode_aggregate: str,
    exclude_diag: bool
) -> Sequence[float]:
    # evaluate all factors
    factor_scores = np.array([
        _eval_factor_fitness(individual, f_idx, f_dist_matrices, factor_sizes=factor_sizes, fitness_mode=fitness_mode, exclude_diag=exclude_diag)
        for f_idx, f_dist_matrices in enumerate(gt_dist_matrices)
    ])
    # aggregate
    return (
        float(np_aggregate(factor_scores[:, 0], mode=obj_mode_aggregate, dtype='float64')),
        float(individual.mean()),
    )


def _eval_factor_fitness(
    individual: np.ndarray,
    f_idx: int,
    f_dist_matrices: np.ndarray,
    factor_sizes: Tuple[int, ...],
    fitness_mode: str,
    exclude_diag: bool,
) -> List[float]:
    # TODO: population should already be shaped
    # TODO: this is inefficient and slow!
    #       ... too many CPU & memory copies
    #       ... convert to numba? or GPU?

    # generate missing mask axis
    f_mask = individual.reshape(factor_sizes)
    f_mask = np.moveaxis(f_mask, f_idx, -1)
    f_mask = f_mask[..., :, None] & f_mask[..., None, :]
    # the diagonal can change statistics
    if exclude_diag:
        diag = np.arange(f_mask.shape[-1])
        f_mask[..., diag, diag] = False
    # mask the distance array | we negate the mask so that TRUE means the item is disabled
    f_dists = np.ma.masked_where(~f_mask, f_dist_matrices)

    # get distances
    if   fitness_mode == 'range':     fitness_sparse = (np.ma.max(f_dists, axis=-1) - np.ma.min(f_dists, axis=-1)).mean()
    elif fitness_mode == 'max':       fitness_sparse = (np.ma.max(f_dists, axis=-1)).mean()
    elif fitness_mode == 'std':       fitness_sparse = (np.ma.std(f_dists, axis=-1)).mean()
    else: raise KeyError(f'invalid fitness_mode: {repr(fitness_mode)}')

    # combined scores
    return [fitness_sparse]


class GlobalMaskDataGA(BooleanMaskGA):
    """
    Optimize a dataset mask for specific properties using a genetic algorithm.

    NOTE: Due to the way deap works, only one instance of this class
          should exist at any one time.
    """

    def __init__(
        self,
        dataset_name: str,
        # objective
        fitness_mode: str = 'range',
        fitness_exclude_diag: bool = True,
        objective_mode_aggregate: str = 'mean',
        objective_mode_weight: float = -1.0,   # minimize range
        objective_size_weight: float = 0.5,  # maximize size
        # population
        population_size: int = 256,
        num_generations: int = 200,
        # mutation
        mate_probability: float = 0.5,    # probability of mating two individuals
        mutate_probability: float = 0.2,  # probability of mutating an individual
        mutate_bit_flip_prob: float = 0.05,
        # job
        n_jobs: int = min(os.cpu_count(), 16),
    ):
        # parameters
        self.fitness_mode             = fitness_mode
        self.fitness_exclude_diag     = fitness_exclude_diag
        self.objective_mode_aggregate = objective_mode_aggregate
        self.objective_mode_weight    = objective_mode_weight
        self.objective_size_weight    = objective_size_weight
        self.dataset_name             = dataset_name
        # load and compute dataset
        gt_data          = H.make_data(dataset_name)
        gt_dist_matrices = cached_compute_all_factor_dist_matrices(dataset_name)
        # initialize
        super().__init__(
            toolbox_new_individual=partial(_toolbox_new_individual, size=len(gt_data)),
            toolbox_eval_individual=partial(_toolbox_eval_individual, gt_dist_matrices=gt_dist_matrices, factor_sizes=gt_data.factor_sizes, fitness_mode=self.fitness_mode, obj_mode_aggregate=self.objective_mode_aggregate, exclude_diag=self.fitness_exclude_diag),
            objective_weights=(self.objective_mode_weight, self.objective_size_weight),
            population_size=population_size,
            num_generations=num_generations,
            mate_probability=mate_probability,
            mutate_probability=mutate_probability,
            mutate_bit_flip_prob=mutate_bit_flip_prob,
            n_jobs=n_jobs,
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return dict(
            dataset_name=self.dataset_name,
            # objective
            fitness_mode=self.fitness_mode,
            objective_mode_aggregate=self.objective_mode_aggregate,
            objective_mode_weight=self.objective_mode_weight,
            objective_size_weight=self.objective_size_weight,
            # population
            population_size=self.population_size,
            num_generations=self.num_generations,
            # mutation
            mate_probability=self.mate_probability,
            mutate_probability=self.mutate_probability,
            mutate_bit_flip_prob=self.mutate_bit_flip_prob,
            # job
            n_jobs=self.n_jobs,
        )


# ========================================================================= #
# ENTRY POINT                                                               #
# ========================================================================= #


def run(
    dataset_name='xysquares_8x8_toy_s4',  # xysquares_8x8_toy_s4, xcolumns_8x_toy_s1
    num_generations: int = 100,
    fitness_mode: str = 'std',
    objective_mode_aggregate: str = 'mean',
    seed_: int = None,
    save: bool = True,
    save_prefix: str = 'TEST',
    n_jobs=min(os.cpu_count(), 16)
):
    # save the starting time for the save path
    time_string = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log.info(f'Starting run at time: {time_string}')

    # determinism
    seed_ = seed_ if (seed_ is not None) else int(np.random.randint(1, 2**31-1))
    seed(seed_)

    # make algorithm
    ga = GlobalMaskDataGA(
        dataset_name=dataset_name,
        # objective
        fitness_mode=fitness_mode,
        fitness_exclude_diag=True,
        objective_mode_aggregate=objective_mode_aggregate,  # min/max seems to be better
        objective_mode_weight=-1.0,  # minimize range
        objective_size_weight=1.0,   # maximize size
        # population
        population_size=128,
        num_generations=num_generations,
        # mutation
        mate_probability=0.5,    # probability of mating two individuals
        mutate_probability=0.5,  # probability of mutating an individual
        mutate_bit_flip_prob=0.05,
        # job
        n_jobs=n_jobs,
    )
    log.info('Final Config' + make_box_str('\n'.join(f'{k}={repr(v)}' for k, v in ga.parameters.items())))

    # run algorithm!
    population, stats, hall_of_fame = ga.fit()

    def individual_ave(individual):
        sub_data = MaskedDataset(
            data=H.make_data(dataset_name, transform_mode='none'),
            mask_or_indices=individual.flatten(),
        )
        print(', '.join(f'{individual.reshape(sub_data._data.factor_sizes).sum(axis=f_idx).mean():2f}' for f_idx in range(sub_data._data.num_factors)))
        # make obs
        ave_obs = np.zeros_like(sub_data[0], dtype='float64')
        for obs in sub_data:
            ave_obs += obs
        return ave_obs / ave_obs.max()

    # plot average images
    H.plt_subplots_imshow(
        [[individual_ave(m) for m in hall_of_fame]],
        col_labels=[f'{np.sum(m)} / {np.prod(m.shape)} |' for m in hall_of_fame],
        title=f'{dataset_name}: g{ga.num_generations} p{ga.population_size} [{ga.fitness_mode}, {ga.objective_mode_aggregate}]',
        show=True, vmin=0.0, vmax=1.0
    )

    # get save path, make parent dir & save!
    if save:
        job_name = f'{(save_prefix + "_" if save_prefix else "")}{ga.dataset_name}_{ga.num_generations}x{ga.population_size}_{ga.fitness_mode}_{ga.objective_mode_aggregate}_{ga.objective_mode_weight}_{ga.objective_size_weight}'
        save_path = ensure_parent_dir_exists(ROOT_DIR, 'out/adversarial_mask', f'{time_string}_{job_name}_mask.npz')
        log.info(f'saving mask data to: {save_path}')
        np.savez(save_path, mask=hall_of_fame[0].copy(), params=ga.parameters, seed=seed_)


ROOT_DIR = os.path.abspath(__file__ + '/../../..')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for objective_mode_aggregate in ['mean']:
        for fitness_mode in ['range']:
            for dataset_name in ['cars3d']:
                print('='*100)
                print(f'[STARTING]: objective_mode_aggregate={repr(objective_mode_aggregate)} fitness_mode={repr(fitness_mode)} dataset_name={repr(dataset_name)}')
                try:
                    run(
                        dataset_name=dataset_name,
                        num_generations=25,
                        seed_=42,
                        save=True,
                        n_jobs=min(os.cpu_count(), 64),
                        fitness_mode=fitness_mode,
                        objective_mode_aggregate=objective_mode_aggregate,
                        save_prefix='RANDOM',
                    )
                except:
                    warnings.warn(f'[FAILED]: objective_mode_aggregate={repr(objective_mode_aggregate)} fitness_mode={repr(fitness_mode)} dataset_name={repr(dataset_name)}')
                print('='*100)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
