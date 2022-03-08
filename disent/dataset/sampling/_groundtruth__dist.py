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

import warnings
from typing import Optional

import numpy as np

from disent.dataset.data import GroundTruthData
from disent.dataset.sampling._base import BaseDisentSampler
from disent.dataset.util.state_space import StateSpace


# ========================================================================= #
# Ground Truth Dist Sampler                                                 #
# ========================================================================= #


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
            # warn about the error below, we need to fix this, but currently it is non-trivial.
            warnings.warn(
                f'Using scaled versions of the distance functions for {repr(self.__class__.__name__)} currently has '
                f'precision errors, sampling may sometimes be incorrect. Care needs to be taken while introducing a '
                f'fix as it is non-trivial, requiring arbitrary precision values. Performance might then be much '
                f'slower too! See: https://shlegeris.com/2018/10/23/sqrt.html and https://cstheory.stackexchange.com/a/4010')
        # checks
        assert triplet_sample_mode in {'random', 'factors', 'manhattan', 'combined'}, 'It is a bug if this fails!'
        assert 0 <= triplet_swap_chance <= 1, 'triplet_swap_chance must be in range [0, 1]'
        # set vars
        self._sample_mode = triplet_sample_mode
        self._swap_chance = triplet_swap_chance
        # dataset variable
        self._state_space: Optional[StateSpace] = None

    def _init(self, dataset):
        assert isinstance(dataset, GroundTruthData), f'dataset must be an instance of {repr(GroundTruthData.__class__.__name__)}, got: {repr(dataset)}'
        self._state_space = dataset.state_space_copy()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _sample_idx(self, idx):
        # sample indices
        indices = (idx, *np.random.randint(0, len(self._state_space), size=self._num_samples-1))
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
        a_f, p_f, n_f = self._state_space.idx_to_pos(indices)
        a_d, p_d, n_d = a_f, p_f, n_f
        # dists vars
        if self._scaled:
            # range of positions is [0, f_size - 1], to scale between 0 and 1 we need to
            # divide by (f_size - 1), but if the factor size is 1, we can't divide by zero
            # so we make the minimum 1.0
            scale = np.maximum(1, np.array(self._state_space.factor_sizes) - 1)
            a_d = a_d / scale
            p_d = p_d / scale
            n_d = n_d / scale
            # TODO: note that there is a major precision error here! this function
            #       can very quickly return incorrect distances! This problem is very similar to that of
            #       "comparing sums of square roots of integers"... because we divide, precision is lost...
            #       in summing we may have equivalent values, but this will return the wrong result!
            #       FIX:
            #       -- we could get around this by using arbitrary precision numerator/denominators
            #          we instead find the LCM over all the `factor_sizes` and multiply accordingly instead of dividing?
            #          so that values are scaled instead of reduced? We can use arbitrary precision decimals to store the
            #          values instead?
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


# ========================================================================= #
# Investigation:                                                            #
# ========================================================================= #


# class Num(object):
#
#     def __init__(self, numerator: int = 0, denominator: int = 1):
#         self.numerator = numerator
#         self.denominator = denominator
#
#     def _convert(self, other):
#         # check values
#         if isinstance(other, int):
#             other = Num(other)
#         assert isinstance(other, Num)
#         # done!
#         return other
#
#     def __truediv__(self, other):
#         other = self._convert(other)
#         # compute values
#         return Num(
#             numerator=self.numerator * other.denominator,
#             denominator=self.denominator * other.numerator,
#         )
#
#     def __mul__(self, other):
#         other = self._convert(other)
#         # compute values
#         return Num(
#             numerator=self.numerator * other.numerator,
#             denominator=self.denominator * other.denominator,
#         )
#
#     def standardise_with(self, other):
#         other = self._convert(other)
#         # compute
#         if self.denominator == other.denominator:
#             return self, other
#         else:
#             # -----------
#             # CONVERT TO THIS:
#             # # get lowest common multiple
#             # # -- always returns a positive number
#             # lcm: int = math.lcm(self.denominator, other.denominator)
#             # # get multiple
#             # self_mul = lcm / self.denominator
#             # other_mul = lcm / other.denominator
#             # assert isinstance(self_mul, int), f'not an int: {repr(self_mul)}, got type: {type(self_mul)}'
#             # assert isinstance(other_mul, int), f'not an int: {repr(other_mul)}, got type: {type(other_mul)}'
#             # # compute values
#             # return Num(
#             #     numerator=self.numerator * self_mul + other.numerator  * other_mul,
#             #     denominator=lcm,
#             # )
#             # -----------
#             # BAD VERSION:
#             # -- this is terrible! it can explode quickly in size...
#             a = Num(numerator=self.numerator * other.denominator, denominator=self.denominator * other.denominator)
#             b = Num(numerator=other.numerator * self.denominator, denominator=self.denominator * other.denominator)
#             assert a.denominator == b.denominator
#             return a, b
#
#     def __add__(self, other):
#         a, b = self.standardise_with(other)
#         return Num(numerator=a.numerator + b.numerator, denominator=a.denominator)
#
#     def __neg__(self):
#         return Num(numerator=-self.numerator, denominator=self.denominator)
#
#     def __pos__(self):
#         return Num(numerator=self.numerator, denominator=self.denominator)
#
#     def __sub__(self, other):
#         return self.__add__(-other)
#
#     def __str__(self):
#         return f'<{self.numerator}/{self.denominator}~={self.numerator/self.denominator}>'
#
#     def __repr__(self):
#         return f'{self.__class__.__name__}(numerator={repr(self.numerator)}, denominator={repr(self.denominator)})'
#
#
#     def __eq__(self, other):
#         a, b = self.standardise_with(other)
#         return a.numerator == b.numerator
#
#     def __lt__(self, other):
#         a, b = self.standardise_with(other)
#         return a.numerator < b.numerator
#
#     def __le__(self, other):
#         a, b = self.standardise_with(other)
#         return a.numerator <= b.numerator
#
#     def __gt__(self, other):
#         a, b = self.standardise_with(other)
#         return a.numerator > b.numerator
#
#     def __ge__(self, other):
#         a, b = self.standardise_with(other)
#         return a.numerator >= b.numerator


# if __name__ == '__main__':
#
#     # def main():
#     #     from disent.dataset import DisentDataset
#     #     from research.code.dataset.data import XYSquaresMinimalData
#     #     from disent.util.seeds import TempNumpySeed
#     #
#     #     def test(sampler, n: int):
#     #         with TempNumpySeed(777):
#     #             dataset = DisentDataset(XYSquaresMinimalData(), sampler=sampler)
#     #             for i in range(n):
#     #                 i = np.random.randint(0, len(dataset))
#     #                 d = dataset[i]
#     #             print()
#     #
#     #     test(GroundTruthDistSampler(3, 'manhattan'),        n=100000)
#     #     test(GroundTruthDistSampler(3, 'manhattan_scaled'), n=100000)
#     #
#     # main()
#
#     from mpmath import mp
#
#     scount, smismatch = 0, 0.
#     Scount, Smismatch = 0, 0.
#
#     progress = tqdm(range(1_000_000))
#
#     for i in progress:
#
#         A, P, N = np.random.randint(0, 8, size=(3, 6))
#
#
#         # scaled = values / 7
#         # # extract
#         # a, p, n = values
#         # sa, sp, sn = scaled
#
#         # compute dists
#         ap = factor_dist(A, P)
#         an = factor_dist(A, N)
#
#         # sap = factor_dist(sa, sp)
#         # san = factor_dist(sa, sn)
#         # Sap = factor_dist_alt(a, p, 7)
#         # San = factor_dist_alt(a, n, 7)
#
#         mp.dps = 1000
#
#         Sap = Num(0)
#         for a, p in zip(A, P):
#             Sap += Num(abs(a - p), 7)
#
#         San = Num(0)
#         for a, n in zip(A, N):
#             San += Num(abs(a - n), 7)
#
#         # get distances
#         if (ap < an) != (Sap < San):
#             scount += 1
#             smismatch = scount / (i+1)
#
#         # get distances
#         # if (ap < an) != (Sap < San):
#         #     Scount += 1
#         #     Smismatch = Scount / (i+1)
#         # update progress
#         progress.set_postfix({'smismatch': smismatch, 'Smismatch': Smismatch})
#
#     # a = np.array([1, 1, 4, 5, 1, 2])
#     # p = np.array([7, 0, 4, 5, 6, 0])
#     # n = np.array([3, 6, 1, 6, 1, 5])
#     #
#     # # sa = np.array([0.14285714, 0.14285714, 0.57142857, 0.71428571, 0.14285714, 0.28571429])
#     # # sp = np.array([1.,         0.,         0.57142857, 0.71428571, 0.85714286, 0.        ])
#     # # sn = np.array([0.42857143, 0.85714286, 0.14285714, 0.85714286, 0.14285714, 0.71428571])
#     #
#     # sa = a / 7
#     # sp = p / 7
#     # sn = n / 7
#     #
#     # print(factor_dist(a, p))
#     # print(factor_dist(a, n))
#     # print(factor_dist(sa, sp))
#     # print(factor_dist(sa, sn))


# ========================================================================= #
# END:                                                                      #
# ========================================================================= #
