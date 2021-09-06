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

from typing import NoReturn

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import research.util as H
from disent.dataset.data import ArrayGroundTruthData
from disent.util.math.random import random_choice_prng


# ========================================================================= #
# Sub Dataset                                                               #
# ========================================================================= #


class SubDataset(Dataset):

    def __init__(self, data: torch.Tensor, mask_or_indices: torch.Tensor):
        assert isinstance(data, torch.Tensor)
        assert isinstance(mask_or_indices, torch.Tensor)
        # check data
        assert not mask_or_indices.dtype.is_floating_point
        # save data
        self._data = data
        # check inputs
        if mask_or_indices.dtype == torch.bool:
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
# Entry Point                                                               #
# ========================================================================= #


def inplace_evolve(mask: torch.Tensor, p_flip=0.01, r_add_remove=0.01) -> NoReturn:
    # FLIP PERCENTAGE OF ACTIVE
    active, total = mask.sum(), len(mask)
    p_flip_active = (p_flip * active) / total
    mask ^= torch.rand(len(mask)) < p_flip_active

    # ADD REMOVE ELEMENTS
    if np.random.random() < 0.5:
        # decrease active elements
        active_idxs = torch.arange(len(mask))[mask]
        disable_idxs = random_choice_prng(active_idxs, size=max(int(len(active_idxs) * r_add_remove), 1), replace=False)
        mask[disable_idxs] = False
    else:
        # increase active elements
        inactive_idxs = torch.arange(len(mask))[mask]
        enable_idxs = random_choice_prng(inactive_idxs, size=max(int(len(inactive_idxs) * r_add_remove), 1), replace=False)
        mask[enable_idxs] = True


# TODO: this is very memory in-efficient!
#      -- change to tensors and index by hand
def evaluate(gt_data, mask: torch.Tensor, batch_size=2048):
    # get subset
    data = SubDataset(data=gt_data.array, mask_or_indices=mask)
    # compute overlap
    idx_a, idx_b = H.pair_indices_random(len(data), approx_batch_size=batch_size)
    deltas = torch.abs(data[idx_a] - data[idx_b]).mean(dim=(-3, -2, -1))
    # compute secondary deltas
    idx_a, idx_b = H.pair_indices_random(len(deltas), approx_batch_size=batch_size * 4)
    loss = torch.abs(deltas[idx_a] - deltas[idx_b]).mean()
    # compute factor differences
    # -- highest score is best!
    return -loss


# from multiprocessing import Pool
#
# POOL = Pool(processes=os.cpu_count())


def evaluate_all(population: torch.Tensor, gt_data: ArrayGroundTruthData, batch_size=2048, threaded=False) -> torch.Tensor:
    if not threaded:
        scores = [
            evaluate(gt_data, mask=mask, batch_size=batch_size)
            for mask in tqdm(population, desc='evaluation')
        ]
    else:
        scores = POOL.starmap(evaluate, iterable=(
            [gt_data, mask, batch_size]
            for mask in population
        ))
    return torch.stack(scores)


@torch.no_grad()
def main(population_size=64, generations=100, p_flip=0.05, r_add_remove=0.05, keep_best=0.25, batch_size=1024*8):
    # get the dataset and delete the transform
    gt_data: ArrayGroundTruthData = H.make_dataset('cars3d', load_into_memory=True).gt_data
    gt_data._transform = None

    # constants
    keep_n = max(int(keep_best * population_size), 1)

    # generate the starting population
    population = torch.ones(population_size, len(gt_data), dtype=torch.bool)
    for mask in population:
        inplace_evolve(mask, p_flip=p_flip, r_add_remove=r_add_remove)

    # repeat for generations
    for g in range(1, generations+1):
        print(f'{"="*100}\nGENERATION: {g}')
        # evaluate population
        # -- highest score is best
        scores = evaluate_all(population, gt_data=gt_data, batch_size=batch_size)
        best_indices = torch.argsort(scores, descending=True)[:keep_n]
        best_scores = scores[best_indices]
        best_population = population[best_indices]
        # log values
        print(scores)
        print(f'[{g:03d}] SCORES: [{", ".join("{:2g}".format(s) for s in best_scores)}]')
        print(f'              [{", ".join("{:4g}".format(m.sum()) for m in best_population)}]')
        # generate children
        child_indices = np.random.choice(keep_n, size=population_size - keep_n, replace=True)
        child_population = torch.clone(best_population[child_indices])
        for child in child_population:
            inplace_evolve(child, p_flip=p_flip, r_add_remove=r_add_remove)
        # generate new population
        population = torch.cat([best_population, child_population], dim=0)

    # evaluate
    scores = evaluate_all(population, gt_data=gt_data, batch_size=batch_size*2)

    # return best
    return population, scores


if __name__ == '__main__':
    pop, scores = main()

    for i, s in enumerate(scores):
        print(f'{i}: {s}')
