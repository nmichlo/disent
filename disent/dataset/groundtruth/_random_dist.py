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

from disent.data.groundtruth import GroundTruthData
from disent.dataset.groundtruth import GroundTruthDataset


class GroundTruthDistDataset(GroundTruthDataset):

    def __init__(
            self,
            ground_truth_data: GroundTruthData,
            transform=None,
            augment=None,
            num_samples=1,
            triplet_sample_mode='manhattan'
    ):
        super().__init__(
            ground_truth_data=ground_truth_data,
            transform=transform,
            augment=augment,
        )
        # checks
        assert num_samples in {1, 2, 3}, f'num_samples ({repr(num_samples)}) must be 1, 2 or 3'
        assert triplet_sample_mode in {'random', 'factors', 'manhattan', 'combined'}, f'sample_mode ({repr(triplet_sample_mode)}) must be one of {["random", "factors", "manhattan", "combined"]}'
        self._num_samples = num_samples
        self._sample_mode = triplet_sample_mode

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def __getitem__(self, idx):
        # sample indices
        indices = (idx, *np.random.randint(0, len(self), size=self._num_samples-1))
        # sort based on mode
        if self._num_samples == 3:
            indices = self._swap_triple(indices)
        # get data
        return self.dataset_get_observation(*indices)

    def _swap_triple(self, indices):
        a_i, p_i, n_i = indices
        a_f, p_f, n_f = self.idx_to_pos(indices)
        # SWAP: factors
        if self._sample_mode == 'factors':
            if factor_diff(a_f, p_f) > factor_diff(a_f, n_f):
                return a_i, n_i, p_i
        # SWAP: manhattan
        elif self._sample_mode == 'manhattan':
            if factor_dist(a_f, p_f) > factor_dist(a_f, n_f):
                return a_i, n_i, p_i
        # SWAP: combined
        elif self._sample_mode == 'combined':
            if factor_diff(a_f, p_f) > factor_diff(a_f, n_f):
                return a_i, n_i, p_i
            elif factor_diff(a_f, p_f) == factor_diff(a_f, n_f):
                if factor_dist(a_f, p_f) > factor_dist(a_f, n_f):
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



