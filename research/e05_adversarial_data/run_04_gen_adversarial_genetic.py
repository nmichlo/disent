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

import multiprocessing
import random
from typing import List

import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scipy.stats import gmean

import research.util as H
from disent.util.seeds import seed
from research.e01_visual_overlap.util_compute_traversal_dists import cached_compute_all_factor_dist_matrices


# ========================================================================= #
# MUTATE                                                                    #
# ========================================================================= #


def cxTwoPointNumpy(ind1, ind2):
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


creator.create("Fitness", base.Fitness, weights=[-1.0])
creator.create("Individual", np.ndarray, fitness=creator.Fitness)


# ========================================================================= #
# BASE BOOLEAN GA                                                           #
# ========================================================================= #


class BooleanMaskGA(object):

    def __init__(
        self,
        population_size: int = 256,
        num_generations: int = 100,
        mate_probability: float = 0.5,    # probability of mating two individuals
        mutate_probability: float = 0.2,  # probability of mutating an individual
        mutate_bit_flip_prob: float = 0.05,
        select_tournament_size: int = 3,
        n_jobs: int = 1,
    ):
        super().__init__()
        self.population_size = population_size
        self.num_generations = num_generations
        self.mate_probability = mate_probability
        self.mutate_probability = mutate_probability
        self.mutate_bit_flip_prob = mutate_bit_flip_prob
        self.select_tournament_size = select_tournament_size
        self.n_jobs = n_jobs

    def _toolbox_new_individual(self):
        raise NotImplementedError

    def _toolbox_eval_individual(self, individual: np.ndarray):
        raise NotImplementedError

    def _create_toolbox(self):
        toolbox = base.Toolbox() # toolbox.__getitem__ = types.MethodType(lambda toolbox, name: getattr(toolbox, name), toolbox)
        # objects
        toolbox.register("individual", self._toolbox_new_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # evaluation
        toolbox.register("evaluate", self._toolbox_eval_individual)
        # mutation
        toolbox.register("mate",   cxTwoPointNumpy)
        toolbox.register("mutate", tools.mutFlipBit, indpb=self.mutate_bit_flip_prob)
        toolbox.register("select", tools.selTournament, tournsize=self.select_tournament_size)
        # workers
        if self._num_workers > 1:
            toolbox.register('map', multiprocessing.Pool(processes=self._num_workers).map)
        # done!
        return toolbox

    @property
    def _num_workers(self):
        if self.n_jobs > 0:   return self.n_jobs
        elif self.n_jobs < 0: return max(multiprocessing.cpu_count() + 1 + self.n_jobs, 1)
        else:                 raise ValueError('`n_jobs == 0` is invalid!')

    def fit(self) -> (List[np.ndarray], tools.Statistics, tools.HallOfFame):
        toolbox = self._create_toolbox()
        # create new population
        population: List[np.ndarray] = toolbox.population(n=self.population_size)
        # allow individuals to be compared
        hall_of_fame = tools.HallOfFame(1, similar=np.array_equal)
        # create statistics tracker
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)
        # run genetic algorithm
        algorithms.eaSimple(
            population=population,
            toolbox=toolbox,
            cxpb=self.mate_probability,
            mutpb=self.mutate_probability,
            ngen=self.num_generations,
            stats=stats,
            halloffame=hall_of_fame,
        )
        # done!
        return population, stats, hall_of_fame


# ========================================================================= #
# MASKING GA                                                                #
# ========================================================================= #


class GlobalMaskDataGA(BooleanMaskGA):

    def __init__(
        self,
        dataset_name: str,
        fitness_mode: str = 'range',
        population_size: int = 256,
        num_generations: int = 100,
        mate_probability: float = 0.5,    # probability of mating two individuals
        mutate_probability: float = 0.2,  # probability of mutating an individual
        mutate_bit_flip_prob: float = 0.05,
        select_tournament_size: int = 3,
        n_jobs: int = 1,
    ):
        super().__init__(
            population_size=population_size,
            num_generations=num_generations,
            mate_probability=mate_probability,
            mutate_probability=mutate_probability,
            mutate_bit_flip_prob=mutate_bit_flip_prob,
            select_tournament_size=select_tournament_size,
            n_jobs=n_jobs,
        )
        # load and compute dataset
        self._gt_data = H.make_data(dataset_name)
        self._all_dist_matrices = cached_compute_all_factor_dist_matrices(dataset_name)
        self._fitness_mode = fitness_mode

    def _toolbox_new_individual(self) -> creator.Individual:
        mask = np.random.randint(0, 2, size=len(self._gt_data), dtype='bool')
        return creator.Individual(mask)

    def _toolbox_eval_individual(self, individual: np.ndarray):
        # TODO: add in extra score to maximize number of elements
        # TODO: fitness function as range of values, and then minimize that range.
        # evaluate all factors
        scores = [
            self._eval_factor(individual, f_idx, f_dist_matrices)
            for f_idx, f_dist_matrices in enumerate(self._all_dist_matrices)
        ]
        # aggregate
        # TODO: could be weird
        # TODO: maybe take max instead of gmean
        return float(gmean(scores, dtype='float64'))

    def _eval_factor(self, individual: np.ndarray, f_idx: int, f_dist_matrices: np.ndarray):
        # generate missing mask axis
        f_mask = individual.reshape(self._gt_data.factor_sizes)
        f_mask = np.moveaxis(f_mask, f_idx, -1)
        f_mask = f_mask[..., :, None] & f_mask[..., None, :]
        # mask array & diagonal
        diag = np.arange(f_mask.shape[-1])
        f_mask[..., diag, diag] = False
        f_dists = np.ma.masked_where(~f_mask, f_dist_matrices)  # TRUE is masked, so we need to negate
        # get distances
        if self._fitness_mode == 'range':
            fitness = (np.ma.max(f_dists, axis=-1) - np.ma.min(f_dists, axis=-1)).mean()
        elif self._fitness_mode == 'std':
            fitness = np.ma.std(f_dists, axis=-1).mean()
        else:
            raise KeyError(f'invalid fitness_mode: {repr(self._fitness_mode)}')
        # done!
        return fitness


# ========================================================================= #
# ENTRY POINT                                                               #
# ========================================================================= #


def main():
    seed(42)
    # run algorithm!
    genetic_algorithm = GlobalMaskDataGA('cars3d')
    genetic_algorithm.fit()


if __name__ == "__main__":
    main()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
