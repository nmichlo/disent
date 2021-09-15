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

import json
import logging
import multiprocessing
import os
import random
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Sequence

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scipy.stats import gmean

import research.util as H
from disent.dataset.wrapper import MaskedDataset
from disent.util.inout.files import AtomicSaveFile
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


# def cxTwoPointNdNumpy(ind1: np.ndarray, ind2: np.ndarray, shape):
#     """
#     Executes a two-point crossover on the input individuals within a hypercube. The two individuals
#     are modified in place and both keep their original length.
#     - Similar to tools.cxTwoPoint but modified for numpy arrays.
#     """
#     shape_orig = ind1.shape
#     assert ind1.shape == ind2.shape
#     # get working shape
#     ind1, ind2 = ind1.reshape(shape), ind2.reshape(shape)
#     # random region
#     ranpoints1 = np.random.randint(0, shape)
#     ranpoints2 = np.random.randint(0, shape)
#     cxpoints1 = np.minimum(ranpoints1, ranpoints2)
#     cxpoints2 = np.maximum(ranpoints1, ranpoints2)
#     sel_tuple = tuple(slice(a, b) for a, b in zip(cxpoints1, cxpoints2))
#     ind1[sel_tuple], ind2[sel_tuple] = ind2[sel_tuple].copy(), ind1[sel_tuple].copy()
#     # restore shape
#     return ind1.__class__(ind1.reshape(shape_orig)), ind2.__class__(ind2.reshape(shape_orig))


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
        self.population_size = population_size
        self.num_generations = num_generations
        self.mate_probability = mate_probability
        self.mutate_probability = mutate_probability
        self.mutate_bit_flip_prob = mutate_bit_flip_prob
        self.n_jobs = n_jobs

    @property
    def parameters(self) -> Dict[str, Any]:
        return dict(
            population_size=self.population_size,
            num_generations=self.num_generations,
            mate_probability=self.mate_probability,
            mutate_probability=self.mutate_probability,
            mutate_bit_flip_prob=self.mutate_bit_flip_prob,
            n_jobs=self.n_jobs,
        )

    def _toolbox_new_individual(self) -> 'creator.Individual':
        raise NotImplementedError

    def _toolbox_eval_individual(self, individual: np.ndarray) -> Sequence[float]:
        raise NotImplementedError

    def _toolbox_pre_create(self):
        raise NotImplementedError

    def _create_toolbox(self):
        self._toolbox_pre_create()
        toolbox = base.Toolbox() # toolbox.__getitem__ = types.MethodType(lambda toolbox, name: getattr(toolbox, name), toolbox)
        # objects
        toolbox.register("individual", self._toolbox_new_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # evaluation
        toolbox.register("evaluate", self._toolbox_eval_individual)
        # mutation
        toolbox.register("mate",   cxTwoPointNumpy)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutate_bit_flip_prob)
        # toolbox.register("select", tools.selTournament, tournsize=self.select_tournament_size)  # does not support multi-objectives
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
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('min',  np.min)
        stats.register('max',  np.max)
        stats.register('mean', np.mean)
        stats.register('std',  np.std)
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
        objective_mode_aggregate: str = 'mean',
        # objective_size_aggregate: str = 'mean',
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
        super().__init__(
            population_size=population_size,
            num_generations=num_generations,
            mate_probability=mate_probability,
            mutate_probability=mutate_probability,
            mutate_bit_flip_prob=mutate_bit_flip_prob,
            n_jobs=n_jobs,
        )
        # objective weights
        self.fitness_mode = fitness_mode
        self.objective_mode_aggregate = objective_mode_aggregate
        # self.objective_size_aggregate = objective_size_aggregate
        self.objective_mode_weight = objective_mode_weight
        self.objective_size_weight = objective_size_weight
        # load and compute dataset
        self.dataset_name = dataset_name
        self._gt_data = H.make_data(dataset_name)
        self._gt_dist_matrices = cached_compute_all_factor_dist_matrices(dataset_name)

    @property
    def parameters(self) -> Dict[str, Any]:
        return dict(
            **super().parameters,
            dataset_name=self.dataset_name,
            fitness_mode=self.fitness_mode,
            objective_mode_aggregate=self.objective_mode_aggregate,
            objective_mode_weight=self.objective_mode_weight,
            objective_size_weight=self.objective_size_weight,
        )

    def _toolbox_new_individual(self) -> 'creator.Individual':
        mask = np.random.randint(0, 2, size=len(self._gt_data), dtype='bool')
        return creator.Individual(mask)

    def _toolbox_eval_individual(self, individual: np.ndarray) -> Sequence[float]:
        # evaluate all factors
        factor_scores = np.array([
            self._eval_factor_fitness(individual, f_idx, f_dist_matrices)
            for f_idx, f_dist_matrices in enumerate(self._gt_dist_matrices)
        ])
        # aggregate
        return (
            float(np_aggregate(factor_scores[:, 0], mode=self.objective_mode_aggregate, dtype='float64')),
            # float(np_aggregate(factor_scores[:, 1], mode=self.objective_size_aggregate, dtype='float64')),
            float(individual.mean()),
        )

    def _toolbox_pre_create(self):
        creator.create("Fitness", base.Fitness, weights=[
            self.objective_mode_weight,
            self.objective_size_weight,
        ])
        creator.create("Individual", np.ndarray, fitness=creator.Fitness)

    def _eval_factor_fitness(self, individual: np.ndarray, f_idx: int, f_dist_matrices: np.ndarray):
        # generate missing mask axis
        f_mask = individual.reshape(self._gt_data.factor_sizes)
        f_mask = np.moveaxis(f_mask, f_idx, -1)
        f_mask = f_mask[..., :, None] & f_mask[..., None, :]

        # maximize dataset size
        # TODO: this might be wrong!
        # tools.DeltaPenalty(feasible, 7.0, distance)
        # fitness_size = np.sum(f_mask, axis=-1, dtype='float32').min(axis=-1).mean()  # maximize elements ... weight dependant
        # fitness_size = np.sum(f_mask, axis=-1, dtype='float64').mean()  # maximize elements ... weight dependant
        # fitness_size = np.mean(f_mask, dtype='float64')  # maximize elements ... weight dependant

        # mask array & diagonal
        diag = np.arange(f_mask.shape[-1])
        f_mask[..., diag, diag] = False
        f_dists = np.ma.masked_where(~f_mask, f_dist_matrices)  # TRUE is ignored, so we need to negate

        # get distances
        if   self.fitness_mode == 'range':     fitness_sparse = (np.ma.max(f_dists, axis=-1) - np.ma.min(f_dists, axis=-1)).mean()
        elif self.fitness_mode == 'range_alt': fitness_sparse = (np.ma.max(f_dists, axis=-1) - np.ma.min(f_dists, axis=-1)).max(axis=-1).mean()
        elif self.fitness_mode == 'max':       fitness_sparse = (np.ma.max(f_dists, axis=-1)).mean()
        elif self.fitness_mode == 'std':       fitness_sparse = (np.ma.std(f_dists, axis=-1)).mean()
        else: raise KeyError(f'invalid fitness_mode: {repr(self._fitness_mode)}')

        # combined scores
        return [fitness_sparse] #, fitness_size]


# ========================================================================= #
# ENTRY POINT                                                               #
# ========================================================================= #


def run(
    dataset_name='xysquares_8x8_toy_s4',  # xysquares_8x8_toy_s4, xcolumns_8x_toy_s2
    num_generations: int = 1000,
    seed_: int = None,
    save: bool = True,
    save_prefix: str = '',
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
        fitness_mode='range',
        objective_mode_aggregate='mean',  # min/max seems to be better
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
        print(', '.join(
            f'{individual.reshape(sub_data._data.factor_sizes).sum(axis=f_idx).mean():2f}'
            for f_idx in range(sub_data._data.num_factors)
        ))
        # make obs
        ave_obs = np.zeros_like(sub_data[0], dtype='float64')
        for obs in sub_data:
            ave_obs += obs
        return ave_obs / ave_obs.max()

    # plot
    H.plt_subplots_imshow(
        [[individual_ave(m) for m in hall_of_fame]],
        col_labels=[f'{np.sum(m)} / {np.prod(m.shape)} |' for m in hall_of_fame],
        title=f'{dataset_name}: g{ga.num_generations} p{ga.population_size} [{ga.fitness_mode}, {ga.objective_mode_aggregate}]',
        show=True, vmin=0.0, vmax=1.0
    )

    # def individual_random(individual, n=1000):
    #     data = H.make_data(dataset_name, transform_mode='none')
    #     # extract data
    #     sub_data = MaskedDataset(
    #         data=data,
    #         mask_or_indices=individual.flatten(),
    #     )
    #     # get random values orig
    #     idxs_a, idxs_b = H.pair_indices_random(len(data), approx_batch_size=n)
    #     deltas_orig = [((data[i] - data[j]) ** 2).mean() for i, j in zip(idxs_a, idxs_b)]
    #     # get random values
    #     idxs_a, idxs_b = H.pair_indices_random(len(sub_data), approx_batch_size=n)
    #     deltas_masked = [((sub_data[i] - sub_data[j]) ** 2).mean() for i, j in zip(idxs_a, idxs_b)]
    #     # done!
    #     return np.array(deltas_masked), np.array(deltas_orig)
    #
    # # plot overlap
    # fig, axs = H.plt_subplots(nrows=1, ncols=len(hall_of_fame) + 1)
    # for i, (m, ax) in enumerate(zip(hall_of_fame, axs.flatten())):
    #     deltas_masked, deltas_orig = individual_random(m, n=5000)
    #     ax.hist(deltas_orig, histtype='step', density=True, cumulative=True)
    #     ax.hist(deltas_masked, histtype='step', density=True, cumulative=True)
    # # random plot
    # deltas_rand, deltas_orig = individual_random(np.random.random(hall_of_fame[0].shape) < np.mean(hall_of_fame[0]), n=5000)
    # axs.flatten()[-1].hist(deltas_orig, histtype='step', density=True, cumulative=True)
    # axs.flatten()[-1].hist(deltas_rand, histtype='step', density=True, cumulative=True)
    # plt.show()

    # get save path, make parent dir & save!
    if save:
        job_name = f'{(save_prefix + "_" if save_prefix else "")}{ga.dataset_name}_{ga.num_generations}x{ga.population_size}_{ga.fitness_mode}_{ga.objective_mode_aggregate}_{ga.objective_mode_weight}_{ga.objective_size_weight}'
        save_path = ensure_parent_dir_exists(ROOT_DIR, 'out/adversarial_mask', f'{time_string}_{job_name}_mask.npz')
        log.info(f'saving mask data to: {save_path}')
        np.savez(save_path, mask=hall_of_fame[0].copy(), params=ga.parameters, seed=seed_)


ROOT_DIR = os.path.abspath(__file__ + '/../../..')


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    for dataset_name in ['smallnorb', 'cars3d', 'shapes3d', 'dsprites']:
        run(dataset_name=dataset_name, num_generations=1000, seed_=42, save=True, n_jobs=1)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
