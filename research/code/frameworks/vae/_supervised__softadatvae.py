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
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.distributions import Normal

from disent.nn.loss.triplet import compute_dist_triplet_loss
from disent.frameworks.vae._supervised__tvae import TripletVae


log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class SoftAdaTripletVae(TripletVae):

    REQUIRED_OBS = 3

    @dataclass
    class cfg(TripletVae.cfg):
        detach_decoder: bool = True
        # ADA VAE
        ada_thresh_ratio: float = 0.5
        # SOFT ADA
        softada_scale_slope: float = 0.
        # TRIPLET LOSS
        triplet_loss: str = 'triplet_soft'
        triplet_margin_min: float = None
        triplet_margin_max: float = None
        triplet_scale: float = 1
        triplet_p: float = 1

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal], zs_sampled: Sequence[torch.Tensor], xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]):
        # get representations
        a_z, p_z, n_z = (d.mean for d in ds_posterior)

        # get deltas [0, 1] ie. tau can be constant, we don't need to compute threshold at: 0.5 * (min + max)
        an_delta = torch.abs(a_z - n_z)                 # (B, Z)
        m = torch.amin(an_delta, dim=-1, keepdim=True)  # (B, 1)
        M = torch.amax(an_delta, dim=-1, keepdim=True)  # (B, 1)
        an_delta_01 = (an_delta - m) / (M - m + 1e-10)  # (B, Z)

        # shift neg values
        # * <  ada_thresh_ratio --- shift towards zero, make it count less for distance
        # * >= ada_thresh_ratio --- leave unchanged
        mask   = an_delta_01 < self.cfg.ada_thresh_ratio    # (B, Z)
        scales = _hyperbolic_scale(x=torch.clamp(an_delta_01 / self.cfg.ada_thresh_ratio, 0, 1), m=self.cfg.softada_scale_slope)
        ones   = torch.ones_like(an_delta_01)

        # generate weight mask
        neg_weights = torch.where(mask, scales, ones)  # with the clamp above, we can actually remove this!

        # compute triplet loss
        return compute_dist_triplet_loss(
            pos_delta = (a_z - p_z),
            neg_delta = (a_z - n_z) * neg_weights,
            cfg=self.cfg,
        )


def _hyperbolic_scale(x, m: float = 0):
    # checks
    # - note that x values should be in the range [0, 1]
    assert -1 <= m <= 1, f'slope constant `m` is out of bounds, must be in range [-1, 1], got: {repr(m)}'
    # handle cases
    if m == 0:
        return x
    else:
        c = 1 / m
        return (x * (1 + c)) / (2 * x - (1 - c))


# def _scale_0_1(values: torch.Tensor, m, M, mode='linear'):
#     return (values - m) / (M - m)


# def _norm_0_1(values: torch.Tensor, dim: int) -> torch.Tensor:
#     m = torch.amin(values, dim=dim, keepdim=True)
#     M = torch.amax(values, dim=dim, keepdim=True)
#     return (values - m) / (M - m)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
