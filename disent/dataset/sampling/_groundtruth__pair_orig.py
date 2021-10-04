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


class GroundTruthPairOrigSampler(BaseDisentSampler):

    def uninit_copy(self) -> 'GroundTruthPairOrigSampler':
        return GroundTruthPairOrigSampler(
            p_k=self.p_k
        )

    def __init__(
            self,
            # num_differing_factors
            p_k: int = 1,
    ):
        """
        Sampler that emulates choosing factors like:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
        """
        super().__init__(num_samples=2)
        # DIFFERING FACTORS
        self.p_k = p_k
        # dataset variable
        self._data: GroundTruthData

    def _init(self, dataset):
        assert isinstance(dataset, GroundTruthData), f'dataset must be an instance of {repr(GroundTruthData.__class__.__name__)}, got: {repr(dataset)}'
        self._data = dataset

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # CORE                                                                  #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _sample_idx(self, idx):
        f0, f1 = self.datapoint_sample_factors_pair(idx)
        return (
            self._data.pos_to_idx(f0),
            self._data.pos_to_idx(f1),
        )

    def datapoint_sample_factors_pair(self, idx):
        """
        This function is based on _sample_weak_pair_factors()
        Except deterministic for the first item in the pair, based off of idx.
        """
        # randomly sample the first observation -- In our case we just use the idx
        sampled_factors = self._data.idx_to_pos(idx)
        # sample the next observation with k differing factors
        next_factors, k = _sample_k_differing(sampled_factors, self._data, k=self.p_k)
        # return the samples
        return sampled_factors, next_factors


def _sample_k_differing(factors, ground_truth_data: GroundTruthData, k=1):
    """
    Resample the factors used for the corresponding item in a pair.
      - Based on simple_dynamics() from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
    """
    # checks for factors
    factors = np.array(factors)
    assert factors.ndim == 1
    # sample k
    if k <= 0:
        k = np.random.randint(1, ground_truth_data.num_factors)
    # randomly choose 1 or k
    # TODO: This is in disentanglement lib, HOWEVER is this not a mistake?
    #       A bug report has been submitted to disentanglement_lib for clarity:
    #       https://github.com/google-research/disentanglement_lib/issues/31
    k = np.random.choice([1, k])
    # generate list of differing indices
    index_list = np.random.choice(len(factors), k, replace=False)
    # randomly update factors
    for index in index_list:
        factors[index] = np.random.choice(ground_truth_data.factor_sizes[index])
    # return!
    return factors, k


def _sample_weak_pair_factors(gt_data: GroundTruthData):  # pragma: no cover
    """
    Sample a weakly supervised pair from the given GroundTruthData.
      - Based on weak_dataset_generator() from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
    """
    # randomly sample the first observation
    sampled_factors = gt_data.sample_factors(1)
    # sample the next observation with k differing factors
    next_factors, k = _sample_k_differing(sampled_factors, gt_data, k=1)
    # return the samples
    return sampled_factors, next_factors
