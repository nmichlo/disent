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

import random
from typing import NoReturn
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.stats import gmean
from torch.utils.data import Dataset
from tqdm import tqdm

import research.e01_visual_overlap.util_compute_traversal_dists as E1
import research.util as H
from disent.util.math.random import random_choice_prng

# ========================================================================= #
# Sub Dataset                                                               #
# ========================================================================= #
from research.util import pair_indices_combinations


class SubDataset(Dataset):

    def __init__(self, data: Sequence, mask_or_indices: Union[torch.Tensor, np.ndarray]):
        assert len(data) == len(mask_or_indices)
        assert mask_or_indices.ndim == 1
        # save data
        self._data = data
        # check inputs
        if mask_or_indices.dtype in ('bool', torch.bool):
            # boolean values
            assert len(data) == len(mask_or_indices)
            self._indices = np.arange(len(data))[mask_or_indices]
        else:
            # integer values
            assert len(np.unique(mask_or_indices)) == len(mask_or_indices)
            self._indices = mask_or_indices

    @property
    def data(self):
        return self._data

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._data[self._indices[idx]]


# ========================================================================= #
# Evolve Helper                                                             #
# ========================================================================= #


def inplace_bit_flip_percent(mask: np.ndarray, p_flip=0.05):
    # FLIP PERCENTAGE OF ACTIVE
    active, total = mask.sum(), np.prod(mask.shape)
    p_flip_active = (p_flip * active) / total
    mask ^= np.random.rand(*mask.shape) < p_flip_active


def inplace_bit_flip(mask: np.ndarray, p_flip=0.05):
    # FLIP PERCENTAGE OF ACTIVE
    mask ^= np.random.rand(*mask.shape) < p_flip


def inplace_crossover(mask0: np.ndarray, mask1: np.ndarray):
    assert mask0.shape == mask1.shape
    # generate indices
    idxs = np.random.randint(0, mask0.shape, size=(2, mask0.ndim))
    idx_m = np.min(idxs, axis=0)
    idx_M = np.max(idxs, axis=0)
    # make slices
    slices = tuple(slice(m, M) for m, M in zip(idx_m, idx_M))
    # perform crossover
    mask0[slices], mask1[slices] = mask1[slices].copy(), mask0[slices].copy()


# def selection_tournament(population, scores, n: int = None, tournament_size: int = 3):
#     assert len(population) == len(scores)
#     # defaults
#     if n is None:
#         n = len(population)
#     # select values
#     chosen_idxs = []
#     while len(chosen_idxs) < n:
#         idxs = random_choice_prng(len(population), size=tournament_size, replace=False)
#         idx = max(idxs, key=lambda i: scores[i])
#         chosen_idxs.append(idx)
#     # return new population
#     return population[chosen_idxs].copy()


def evolved_population_inplace(
    population: Sequence[np.ndarray],
    scores: Sequence[float] = None,
    #
    p_mutate: float = 0.2,
    p_mutate_flip: float = 0.05,
    #
    p_crossover: float = 0.5,
) -> NoReturn:
    # SELECTION TOURNAMENT
    # if scores is not None:
    #     population = selection_tournament(population, scores)
    # FLIP BITS
    for mask in population:
        if random.random() < p_mutate:
            inplace_bit_flip(mask, p_flip=p_mutate_flip)
    # CROSSOVER
    indices = np.arange(len(population))
    np.random.shuffle(indices)
    for i in range(1, len(population), 2):
        if random.random() < p_crossover:
            inplace_crossover(population[indices[i - 1]], population[indices[i]])
    # done!
    return population.copy()


# ========================================================================= #
# Evaluation                                                                #
# ========================================================================= #


def evaluate_all(population: np.ndarray, all_dist_matrices, mode: str = 'maximize') -> (np.ndarray, np.ndarray):
    scores = np.array([
        evaluate(all_dist_matrices, mask=mask)
        for mask in population
    ])
    # handle mode
    if   mode == 'maximize': indices = np.argsort(scores)[::-1]
    elif mode == 'minimize': indices = np.argsort(scores)
    else: raise KeyError(f'invalid mode: {repr(mode)}')
    # return sorted population
    return population[indices], scores[indices]


def evaluate(all_dist_matrices, mask: np.ndarray) -> float:
    # evaluate all factors
    scores = []
    for f_idx, f_dist_matrices in enumerate(all_dist_matrices):
        f_mask = np.moveaxis(mask, f_idx, -1)
        f_mask = f_mask[..., :, None] & f_mask[..., None, :]

        # (X, Y, Y)
        # (Y, X, X)

        # mask array & diagonal
        diag = np.arange(f_mask.shape[-1])
        f_mask[..., diag, diag] = False
        f_dists = np.ma.masked_where(~f_mask, f_dist_matrices)  # TRUE is masked, so we need to negate

        # compute score
        # fitness = np.ma.std(f_dists, axis=-1).mean()
        fitness = (np.ma.max(f_dists, axis=-1) - np.ma.min(f_dists, axis=-1)).mean()

        # TODO: add in extra score to maximize number of elements
        # TODO: fitness function as range of values, and then minimize that range.

        # final
        scores.append(fitness)

    # final score
    # TODO: could be weird
    # TODO: maybe take max instead of gmean
    return float(gmean(scores, dtype='float64'))


# ========================================================================= #
# Evolutionary Algorithm                                                    #
# ========================================================================= #


def overlap_genetic_algorithm(
    dataset_name: str,
    population_size=256,
    generations=200,
    mode: str = 'minimize',
    keep_best_ratio: float = 0.2,
):
    # make dataset & precompute distances
    gt_data = H.make_data(dataset_name, transform_mode='float32')
    E1.print_dist_matrix_stats(gt_data)
    all_dist_matrices = E1.cached_compute_all_factor_dist_matrices(dataset_name, compute_workers=64, compute_batch_size=32, force=False)

    # get ratio
    keep_n_best = max(1, int(population_size * keep_best_ratio))

    # generate the starting population
    population = np.ones([population_size, *gt_data.factor_sizes], dtype='bool')
    # population = np.random.randint(0, 2, size=[population_size, *gt_data.factor_sizes], dtype='bool')
    # population[0, :] = False
    # population[0, 0] = True
    # population[0, 4] = True


    # TODO: 1. pairwise distances over all pairs
    #

    # TODO: investigate why trivial solution with only 1 or 2
    #       elements is not happening.

    p = tqdm(total=generations, desc='generation', ncols=150)
    # repeat for generations
    for g in range(1, generations + 1):
        # evaluate population
        population, scores = evaluate_all(population, all_dist_matrices=all_dist_matrices, mode=mode)
        # evolve population
        population = np.concatenate([
            population[:keep_n_best],
            evolved_population_inplace(population[:-keep_n_best], scores[:-keep_n_best])
        ])
        assert len(population) == population_size
        # inject
        # population[0, :] = False
        # population[0, 0] = True
        # population[0, 4] = True
        # update progress bar
        p.update()
        scores = {
            'min': scores.min(),
            'max': scores.max(),
            'mean': scores.mean(),
            'std': scores.std(),
            'best_score':  scores[0],
            'best_active': population[0].sum(),
            'best_total':  population[0].size,
        }
        p.set_postfix(scores)
        print(scores)
        print(population[0].astype('int').tolist())

    # evaluate
    population, scores = evaluate_all(population, all_dist_matrices=all_dist_matrices, mode=mode)
    # return best
    return population, scores


# ========================================================================= #
# Entry Point                                                               #
# ========================================================================= #


if __name__ == '__main__':
    # optimize!
    # dataset_name = 'xcolumns_8x_toy_s2'
    dataset_name = 'xysquares_8x8_toy_s4'
    # dataset_name = 'cars3d'
    population, scores = overlap_genetic_algorithm(dataset_name=dataset_name, population_size=128, generations=500)

    # population = np.array([[]])

    # extract data
    sub_data = SubDataset(
        data=H.make_data(dataset_name, transform_mode='none'),
        mask_or_indices=population[0].flatten(),
    )

    # make obs
    ave_obs = np.zeros_like(sub_data[0], dtype='float64')
    for obs in sub_data:
        ave_obs += obs
    plt.imshow(ave_obs / ave_obs.max())
    plt.show()



# ========================================================================= #
# END                                                                       #
# ========================================================================= #
