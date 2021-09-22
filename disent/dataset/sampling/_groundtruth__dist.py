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

import numpy as np

from disent.dataset.data import GroundTruthData
from disent.dataset.sampling._base import BaseDisentSampler


class GroundTruthDistSampler(BaseDisentSampler):

    def uninit_copy(self) -> 'GroundTruthDistSampler':
        return GroundTruthDistSampler(
            num_samples=self._num_samples,
            triplet_sample_mode=self._triplet_sample_mode,
            triplet_swap_chance=self._triplet_swap_chance,
        )

    def __init__(
            self,
            num_samples=1,
            triplet_sample_mode='manhattan_scaled',
            triplet_swap_chance=0.0,
    ):
        super().__init__(num_samples=num_samples)
        # checks
        assert num_samples in {1, 2, 3}, f'num_samples ({repr(num_samples)}) must be 1, 2 or 3'
        assert triplet_sample_mode in {'random', 'factors', 'manhattan', 'manhattan_scaled', 'combined', 'combined_scaled'}, f'sample_mode ({repr(triplet_sample_mode)}) must be one of {["random", "factors", "manhattan", "combined"]}'
        # save hparams
        self._num_samples = num_samples
        self._triplet_sample_mode = triplet_sample_mode
        self._triplet_swap_chance = triplet_swap_chance
        # scaled
        self._scaled = False
        if triplet_sample_mode.endswith('_scaled'):
            triplet_sample_mode = triplet_sample_mode[:-len('_scaled')]
            self._scaled = True
        # checks
        assert triplet_sample_mode in {'random', 'factors', 'manhattan', 'combined'}, 'It is a bug if this fails!'
        assert 0 <= triplet_swap_chance <= 1, 'triplet_swap_chance must be in range [0, 1]'
        # set vars
        self._sample_mode = triplet_sample_mode
        self._swap_chance = triplet_swap_chance
        # dataset variable
        self._data: GroundTruthData

    def _init(self, dataset):
        assert isinstance(dataset, GroundTruthData), f'dataset must be an instance of {repr(GroundTruthData.__class__.__name__)}, got: {repr(dataset)}'
        self._data = dataset

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _sample_idx(self, idx):
        # sample indices
        indices = (idx, *np.random.randint(0, len(self._data), size=self._num_samples-1))
        # sort based on mode
        if self._num_samples == 3:
            a_i, p_i, n_i = self._swap_triple(indices)
            # randomly swap positive and negative
            if np.random.random() < self._swap_chance:
                indices = (a_i, n_i, p_i)
            else:
                indices = (a_i, p_i, n_i)
        # get data
        return indices

    def _swap_triple(self, indices):
        a_i, p_i, n_i = indices
        a_f, p_f, n_f = self._data.idx_to_pos(indices)
        a_d, p_d, n_d = a_f, p_f, n_f
        # dists vars
        if self._scaled:
            # range of positions is [0, f_size - 1], to scale between 0 and 1 we need to
            # divide by (f_size - 1), but if the factor size is 1, we can't divide by zero
            # so we make the minimum 1.0
            scale = np.maximum(1, np.array(self._data.factor_sizes) - 1)
            a_d = a_d / scale
            p_d = p_d / scale
            n_d = n_d / scale
        # SWAP: factors
        if self._sample_mode == 'factors':
            if factor_diff(a_f, p_f) > factor_diff(a_f, n_f):
                return a_i, n_i, p_i
        # SWAP: manhattan
        elif self._sample_mode == 'manhattan':
            if factor_dist(a_d, p_d) > factor_dist(a_d, n_d):
                return a_i, n_i, p_i
        # SWAP: combined
        elif self._sample_mode == 'combined':
            if factor_diff(a_f, p_f) > factor_diff(a_f, n_f):
                return a_i, n_i, p_i
            elif factor_diff(a_f, p_f) == factor_diff(a_f, n_f):
                if factor_dist(a_d, p_d) > factor_dist(a_d, n_d):
                    return a_i, n_i, p_i
        # SWAP: random
        elif self._sample_mode != 'random':
            raise KeyError('invalid mode')
        # done!
        return indices


def factor_diff(f0, f1):
    return np.sum(f0 != f1)


def factor_dist(f0, f1):
    return np.sum(np.abs(f0 - f1))
