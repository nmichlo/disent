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


import logging
import warnings
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch

import experiment.exp.util as H
from disent.dataset.data import GroundTruthData
from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.sampling import GroundTruthPairSampler
from disent.dataset.sampling import GroundTruthTripleSampler
from disent.nn.loss.reduction import batch_loss_reduction
from disent.util.strings import colors as c
from experiment.exp.util import unreduced_loss


log = logging.getLogger(__name__)


# ========================================================================= #
# Samplers                                                                  #
# ========================================================================= #


class AdversarialSampler_CloseFar(BaseDisentSampler):

    def __init__(
        self,
        close_p_k_range=(1, 1),
        close_p_radius_range=(1, 1),
        far_p_k_range=(1, -1),
        far_p_radius_range=(1, -1),
    ):
        super().__init__(3)
        self.sampler_close = GroundTruthPairSampler(p_k_range=close_p_k_range, p_radius_range=close_p_radius_range)
        self.sampler_far = GroundTruthPairSampler(p_k_range=far_p_k_range, p_radius_range=far_p_radius_range)

    def _init(self, gt_data: GroundTruthData):
        self.sampler_close.init(gt_data)
        self.sampler_far.init(gt_data)

    def _sample_idx(self, idx: int) -> Tuple[int, ...]:
        # sample indices
        anchor, pos = self.sampler_close(idx)
        _anchor, neg = self.sampler_far(idx)
        assert anchor == _anchor
        # return triple
        return anchor, pos, neg


class AdversarialSampler_SameK(BaseDisentSampler):

    def __init__(self, k: Union[Literal['random'], int] = 'random', sample_p_close: bool = False):
        super().__init__(3)
        self._gt_data: GroundTruthData = None
        self._sample_p_close = sample_p_close
        self._k = k
        assert (isinstance(k, int) and k > 0) or (k == 'random')

    def _init(self, gt_data: GroundTruthData):
        self._gt_data = gt_data

    def _sample_idx(self, idx: int) -> Tuple[int, ...]:
        a_factors = self._gt_data.idx_to_pos(idx)
        # SAMPLE FACTOR INDICES
        k = self._k
        if k == 'random':
            k = np.random.randint(1, self._gt_data.num_factors+1)  # end exclusive, ie. [1, num_factors+1)
        # get shared mask
        shared_indices = np.random.choice(self._gt_data.num_factors, size=self._gt_data.num_factors-k, replace=False)
        shared_mask = np.zeros(a_factors.shape, dtype='bool')
        shared_mask[shared_indices] = True
        # generate values
        p_factors = self._sample_shared(a_factors, shared_mask, sample_close=self._sample_p_close)
        n_factors = self._sample_shared(a_factors, shared_mask, sample_close=False)
        # swap values if wrong
        # TODO: this might give errors!
        #       - one factor might be less than another
        if np.sum(np.abs(a_factors - p_factors)) > np.sum(np.abs(a_factors - n_factors)):
            p_factors, n_factors = n_factors, p_factors
        # check values
        assert np.sum(a_factors != p_factors) == k, 'this should never happen!'
        assert np.sum(a_factors != n_factors) == k, 'this should never happen!'
        # return values
        return tuple(self._gt_data.pos_to_idx([
            a_factors,
            p_factors,
            n_factors,
        ]))

    def _sample_shared(self, base_factors, shared_mask, tries=100, sample_close: bool = False):
        sampled_factors = base_factors.copy()
        generate_mask = ~shared_mask
        # generate values
        for i in range(tries):
            if sample_close:
                sampled_values = (base_factors + np.random.randint(-1, 1+1, size=self._gt_data.num_factors))
                sampled_values = np.clip(sampled_values, 0, np.array(self._gt_data.factor_sizes) - 1)[generate_mask]
            else:
                sampled_values = np.random.randint(0, np.array(self._gt_data.factor_sizes)[generate_mask])
            # overwrite values that are not different
            sampled_factors[generate_mask] = sampled_values
            # update mask
            sampled_shared_mask = (sampled_factors == base_factors)
            generate_mask &= sampled_shared_mask
            # check everything
            if np.sum(sampled_shared_mask) == np.sum(shared_mask):
                assert np.sum(generate_mask) == 0
                return sampled_factors
            # we need to try again!
        raise RuntimeError('could not generate factors: {}')


def sampler_print_test(sampler: Union[str, BaseDisentSampler], gt_data: GroundTruthData = None, steps=100):
    # make data
    if gt_data is None:
        gt_data = H.make_dataset('xysquares_8x8_mini').gt_data
    # make sampler
    if isinstance(sampler, str):
        prefix = sampler
        sampler = make_adversarial_sampler(sampler)
    else:
        prefix = sampler.__class__.__name__
    if not sampler.is_init:
        sampler.init(gt_data)
    # print everything
    count_pn_k0, count_pn_d0 = 0, 0
    for i in range(min(steps, len(gt_data))):
        a, p, n = gt_data.idx_to_pos(sampler(i))
        ap_k = np.sum(a != p); ap_d = np.sum(np.abs(a - p))
        an_k = np.sum(a != n); an_d = np.sum(np.abs(a - n))
        pn_k = np.sum(p != n); pn_d = np.sum(np.abs(p - n))
        print(f'{prefix}: [{c.lGRN}ap{c.RST}:{ap_k:2d}:{ap_d:2d}] [{c.lRED}an{c.RST}:{an_k:2d}:{an_d:2d}] [{c.lYLW}pn{c.RST}:{pn_k:2d}:{pn_d:2d}] {a} {p} {n}')
        count_pn_k0 += (pn_k == 0)
        count_pn_d0 += (pn_d == 0)
    print(f'count pn:(k=0) = {count_pn_k0} pn:(d=0) = {count_pn_d0}')


def make_adversarial_sampler(mode: str = 'close_far'):
    if mode == 'close_far':
        return AdversarialSampler_CloseFar(
            close_p_k_range=(1, 1), close_p_radius_range=(1, 1),
            far_p_k_range=(0, -1), far_p_radius_range=(0, -1),
        )
    elif mode == 'same_k':
        return AdversarialSampler_SameK(k='random', sample_p_close=False)
    elif mode == 'same_k_close':
        return AdversarialSampler_SameK(k='random', sample_p_close=True)
    elif mode == 'same_k1_close':
        return AdversarialSampler_SameK(k=1, sample_p_close=True)
    elif mode == 'close_factor_far_random':
        return GroundTruthTripleSampler(
            p_k_range=(1, 1), n_k_range=(1, -1), n_k_sample_mode='bounded_below', n_k_is_shared=True,
            p_radius_range=(1, -1), n_radius_range=(0, -1), n_radius_sample_mode='bounded_below',
        )
    elif mode == 'close_far_same_factor':
        # TODO: problematic for dsprites
        return GroundTruthTripleSampler(
            p_k_range=(1, 1), n_k_range=(1, 1), n_k_sample_mode='bounded_below', n_k_is_shared=True,
            p_radius_range=(1, 1), n_radius_range=(2, -1), n_radius_sample_mode='bounded_below',
        )
    elif mode == 'same_factor':
        return GroundTruthTripleSampler(
            p_k_range=(1, 1), n_k_range=(1, 1), n_k_sample_mode='bounded_below', n_k_is_shared=True,
            p_radius_range=(1, -2), n_radius_range=(2, -1), n_radius_sample_mode='bounded_below',  # bounded below does not always work, still relies on random chance :/
        )
    elif mode == 'random_bb':
        return GroundTruthTripleSampler(
            p_k_range=(0, -1), n_k_range=(0, -1), n_k_sample_mode='bounded_below', n_k_is_shared=True,
            p_radius_range=(0, -1), n_radius_range=(0, -1), n_radius_sample_mode='bounded_below',
        )
    elif mode == 'random_swap_manhat':
        return GroundTruthTripleSampler(
            p_k_range=(0, -1), n_k_range=(0, -1), n_k_sample_mode='random', n_k_is_shared=False,
            p_radius_range=(0, -1), n_radius_range=(0, -1), n_radius_sample_mode='random',
            swap_metric='manhattan'
        )
    elif mode == 'random_swap_manhat_norm':
        return GroundTruthTripleSampler(
            p_k_range=(0, -1), n_k_range=(0, -1), n_k_sample_mode='random', n_k_is_shared=False,
            p_radius_range=(0, -1), n_radius_range=(0, -1), n_radius_sample_mode='random',
            swap_metric='manhattan_norm'
        )
    else:
        raise KeyError(f'invalid adversarial sampler: mode={repr(mode)}')


# ========================================================================= #
# Adversarial Loss                                                          #
# ========================================================================= #

# anchor, positive, negative
TensorTriple = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _get_triple(x: TensorTriple, adversarial_swapped: bool):
    if not adversarial_swapped:
        a, p, n = x
    else:
        a, n, p = x
    return a, p, n


def adversarial_loss(
    ys: TensorTriple,
    # adversarial loss settings
    adversarial_mode: str = 'invert_shift',
    adversarial_swapped: bool = False,
    adversarial_const_target: Optional[float] = None,  # only used if loss_mode=="const"
    # pixel loss to get deltas settings
    pixel_loss_mode: str = 'mse',
):
    a_y, p_y, n_y = _get_triple(ys, adversarial_swapped=adversarial_swapped)

    # compute deltas
    p_deltas = H.pairwise_loss(a_y, p_y, mode=pixel_loss_mode, mean_dtype=torch.float32)
    n_deltas = H.pairwise_loss(a_y, n_y, mode=pixel_loss_mode, mean_dtype=torch.float32)

    # compute loss
    if adversarial_mode == 'const':
        # check values
        if not isinstance(adversarial_const_target, (int, float)):
            raise ValueError(f'loss_mode=="const" requires a numerical value for `adversarial_const_target`, got: {repr(adversarial_const_target)}')
        # compute loss
        p_loss = torch.abs(adversarial_const_target - p_deltas).mean()  # should this be l2 dist instead?
        n_loss = torch.abs(adversarial_const_target - n_deltas).mean()  # should this be l2 dist instead?
        return p_loss + n_loss

    # check values
    if adversarial_const_target is not None:
        warnings.warn(f'`adversarial_loss` only supports a value for `adversarial_const_target` when `loss_mode=="const"`')

    # deltas
    deltas = (n_deltas - p_deltas)

    # compute loss
    if   adversarial_mode == 'self':             return torch.abs(deltas).mean()  # should this be l2 dist instead?
    elif adversarial_mode == 'invert_unbounded': return deltas.mean()
    elif adversarial_mode == 'invert':           return torch.maximum(deltas, torch.zeros_like(deltas)).mean()
    elif adversarial_mode == 'invert_shift':     return torch.maximum(0.01 + deltas, torch.zeros_like(deltas)).mean()  # triplet_loss = torch.clamp_min(p_dist - n_dist + margin_max, 0)
    else:
        raise KeyError(f'invalid `adversarial_mode`: {repr(adversarial_mode)}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


# if __name__ == '__main__':
#     sampler_print_test(
#         sampler='same_k',
#         gt_data=XYObjectData()
#     )
