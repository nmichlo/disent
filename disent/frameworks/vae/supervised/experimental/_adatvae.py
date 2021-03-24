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
from dataclasses import dataclass
from dataclasses import field
from typing import Mapping
from typing import Sequence

import torch
import numpy as np
from torch.distributions import Normal

from disent.frameworks.helper.triplet_loss import configured_triplet, configured_dist_triplet
from disent.frameworks.vae.unsupervised import Vae
from disent.frameworks.vae.supervised import TripletVae
from disent.frameworks.vae.weaklysupervised import AdaVae
import logging

from disent.schedule import CyclicSchedule
from disent.schedule import Schedule
from experiment.util.hydra_utils import instantiate_recursive
from experiment.util.hydra_utils import make_target_dict


log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AdaTripletVae(TripletVae):

    REQUIRED_OBS = 3

    @dataclass
    class cfg(TripletVae.cfg):
        # adatvae: what version of triplet to use
        triplet_mode: str = 'trip_AND_softAve'
        # adatvae: annealing
        ada_triplet_ratio: float = 0.5
        # TODO: experiment runner does not support referencing config variables if target is defined in a config
        ada_triplet_schedule: dict = field(default_factory=lambda: make_target_dict(
            CyclicSchedule,
            period=3600,
            mode='linear',
        ))

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # register schedule
        if self.cfg.ada_triplet_schedule is not None:
            if isinstance(self.cfg.ada_triplet_schedule, Mapping):
                self.register_schedule(
                    target='ada_triplet_ratio',
                    schedule=instantiate_recursive(self.cfg.ada_triplet_schedule)
                )
            else:
                assert isinstance(self.cfg.ada_triplet_schedule, Schedule), f'cfg.ada_triplet_schedule is not a hydra dict _target_ or Schedule instance: {repr(self.cfg.ada_triplet_schedule)}'
        # warn if schedule does not exist! this is core to this method!
        if not self.has_schedule('ada_triplet_ratio'):
            warnings.warn(f'{self.__class__.__name__} has no schedule for ada_triplet_ratio')

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal], zs_sampled: Sequence[torch.Tensor], xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]):
        return self.compute_ada_triplet_loss(zs_mean=[d.mean for d in ds_posterior], cfg=self.cfg)

    @staticmethod
    def compute_ada_triplet_loss(zs_mean: Sequence[Normal], cfg: cfg):
        a_z_mean, p_z_mean, n_z_mean = zs_mean

        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # normal triplet
        trip_loss = configured_triplet(a_z_mean, p_z_mean, n_z_mean, cfg=cfg)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # Hard Averaging Before Triplet
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        _, _, an_a_ave, an_n_ave = AdaTripletVae.compute_ave(a_z_mean, p_z_mean, n_z_mean, lerp=None)
        triplet_hard_ave_neg = configured_dist_triplet(pos_delta=p_z_mean-a_z_mean, neg_delta=an_n_ave-an_a_ave, cfg=cfg)  # TODO: good reason why `ap_p_ave - ap_a_ave` this is bad?
        # - - - - - - - - - - - - - - - - #
        _, _, an_a_ave, an_n_ave = AdaTripletVae.compute_ave(a_z_mean, p_z_mean, n_z_mean, lerp=cfg.ada_triplet_ratio)
        triplet_hard_ave_neg_lerp = configured_dist_triplet(pos_delta=p_z_mean-a_z_mean, neg_delta=an_n_ave-an_a_ave, cfg=cfg)  # TODO: good reason why `ap_p_ave - ap_a_ave` this is bad?
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # MSE Averaging With Triplet Loss
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # TODO: this should be scaled separately from triplet?
        soft_ave_loss      = AdaTripletVae.compute_shared_loss(a_z_mean, p_z_mean, n_z_mean, lerp=None) * cfg.triplet_scale
        soft_ave_loss_lerp = AdaTripletVae.compute_shared_loss(a_z_mean, p_z_mean, n_z_mean, lerp=cfg.ada_triplet_ratio) * cfg.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # Scale Triplet Deltas
        # - Triplet but multiply the shared deltas elements so they are
        #   moved closer together. ie. 2x for a->p, and 0.5x for a->n
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        p_shared_mask, n_shared_mask = AdaTripletVae.compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=None)
        p_shared_mask_lerp, n_shared_mask_lerp = AdaTripletVae.compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=cfg.ada_triplet_ratio)
        # - - - - - - - - - - - - - - - - #
        p_mul = torch.where(p_shared_mask, torch.full_like(a_z_mean, 2.0), torch.full_like(a_z_mean, 1.0))
        n_div = torch.where(n_shared_mask, torch.full_like(a_z_mean, 2.0), torch.full_like(a_z_mean, 1.0))
        ada_mul_triplet = configured_dist_triplet(pos_delta=(p_z_mean-a_z_mean)*p_mul, neg_delta=(n_z_mean-a_z_mean)/n_div, cfg=cfg)
        ada_mul_triplet_OLD = configured_dist_triplet(pos_delta=(p_z_mean-a_z_mean)*p_mul, neg_delta=(n_z_mean-a_z_mean)/p_mul, cfg=cfg)
        # - - - - - - - - - - - - - - - - #
        p_mul = torch.where(p_shared_mask, torch.full_like(a_z_mean, 1.0+cfg.ada_triplet_ratio), torch.full_like(a_z_mean, 1.0))
        n_div = torch.where(n_shared_mask, torch.full_like(a_z_mean, 1.0+cfg.ada_triplet_ratio), torch.full_like(a_z_mean, 1.0))
        ada_mul_lerp_triplet = configured_dist_triplet(pos_delta=(p_z_mean-a_z_mean)*p_mul, neg_delta=(n_z_mean-a_z_mean)/n_div, cfg=cfg)
        ada_mul_lerp_triplet_OLD = configured_dist_triplet(pos_delta=(p_z_mean-a_z_mean)*p_mul, neg_delta=(n_z_mean-a_z_mean)/p_mul, cfg=cfg)
        # - - - - - - - - - - - - - - - - #
        p_mul = torch.where(p_shared_mask_lerp, torch.full_like(a_z_mean, 2.0), torch.full_like(a_z_mean, 1.0))
        n_div = torch.where(n_shared_mask_lerp, torch.full_like(a_z_mean, 2.0), torch.full_like(a_z_mean, 1.0))
        ada_lerp_mul_triplet = configured_dist_triplet(pos_delta=(p_z_mean-a_z_mean)*p_mul, neg_delta=(n_z_mean-a_z_mean)/n_div, cfg=cfg)
        ada_lerp_mul_triplet_OLD = configured_dist_triplet(pos_delta=(p_z_mean-a_z_mean)*p_mul, neg_delta=(n_z_mean-a_z_mean)/p_mul, cfg=cfg)
        # - - - - - - - - - - - - - - - - #
        p_mul = torch.where(p_shared_mask_lerp, torch.full_like(a_z_mean, 1.0+cfg.ada_triplet_ratio), torch.full_like(a_z_mean, 1.0))
        n_div = torch.where(n_shared_mask_lerp, torch.full_like(a_z_mean, 1.0+cfg.ada_triplet_ratio), torch.full_like(a_z_mean, 1.0))
        ada_lerp_mul_lerp_triplet = configured_dist_triplet(pos_delta=(p_z_mean-a_z_mean)*p_mul, neg_delta=(n_z_mean-a_z_mean)/n_div, cfg=cfg)
        ada_lerp_mul_lerp_triplet_OLD = configured_dist_triplet(pos_delta=(p_z_mean-a_z_mean)*p_mul, neg_delta=(n_z_mean-a_z_mean)/p_mul, cfg=cfg)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        losses = {
            # normal
            'triplet': trip_loss,
            # Hard Averaging Before Triplet
                          'trip_hardAveNeg':     triplet_hard_ave_neg,
                          'trip_hardAveNegLerp': triplet_hard_ave_neg_lerp,  # GOOD
                  'trip_TO_trip_hardAveNeg':     blend(trip_loss, triplet_hard_ave_neg,      alpha=cfg.ada_triplet_ratio),
                  'trip_TO_trip_hardAveNegLerp': blend(trip_loss, triplet_hard_ave_neg_lerp, alpha=cfg.ada_triplet_ratio),
            'CONST_trip_TO_trip_hardAveNeg':     blend(trip_loss, triplet_hard_ave_neg,      alpha=0.5),
            'CONST_trip_TO_trip_hardAveNegLerp': blend(trip_loss, triplet_hard_ave_neg_lerp, alpha=0.5),
            # MSE Averaging With Triplet Loss
                  'trip_AND_softAve':     trip_loss + (soft_ave_loss      * cfg.ada_triplet_ratio),
                  'trip_AND_softAveLerp': trip_loss + (soft_ave_loss_lerp * cfg.ada_triplet_ratio),
            'CONST_trip_AND_softAve':     trip_loss + soft_ave_loss,
            'CONST_trip_AND_softAveLerp': trip_loss + soft_ave_loss_lerp,
            # Scale Triplet Deltas
                  'trip_scaleAve':      ada_mul_lerp_triplet,
                  'trip_scaleAveLerp':  ada_lerp_mul_lerp_triplet,
            'CONST_trip_scaleAve':      ada_mul_triplet,
            'CONST_trip_scaleAveLerp':  ada_lerp_mul_triplet,
            # BROKEN -- same as scale triplet deltas, but negative is divided by positive share mask...
                  'BROKEN_trip_scaleAve':     ada_mul_lerp_triplet_OLD,
                  'BROKEN_trip_scaleAveLerp': ada_lerp_mul_lerp_triplet_OLD,
            'CONST_BROKEN_trip_scaleAve':     ada_mul_triplet_OLD,
            'CONST_BROKEN_trip_scaleAveLerp': ada_lerp_mul_triplet_OLD,
        }

        return losses[cfg.triplet_mode], {
            **losses,
            'triplet_chosen': losses[cfg.triplet_mode],
            # shared
            'p_shared':      p_shared_mask.sum(dim=1).float().mean(),
            'n_shared':      n_shared_mask.sum(dim=1).float().mean(),
            'p_shared_lerp': p_shared_mask_lerp.sum(dim=1).float().mean(),
            'n_shared_lerp': n_shared_mask_lerp.sum(dim=1).float().mean(),
        }

    @staticmethod
    def compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=None):
        # ADAPTIVE COMPONENT
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        delta_p = torch.abs(a_z_mean - p_z_mean)
        delta_n = torch.abs(a_z_mean - n_z_mean)
        # get thresholds
        p_thresh = AdaVae.estimate_threshold(delta_p, keepdim=True)
        n_thresh = AdaVae.estimate_threshold(delta_n, keepdim=True)
        # interpolate threshold
        if lerp is not None:
            p_thresh = blend(torch.min(delta_p, dim=-1, keepdim=True).values, p_thresh, alpha=lerp)
            n_thresh = blend(torch.min(delta_n, dim=-1, keepdim=True).values, n_thresh, alpha=lerp)
            # -------------- #
            # # RANDOM LERP:
            # # This should average out to the value given above
            # p_min, p_max = torch.min(delta_p, dim=-1, keepdim=True).values, torch.max(delta_p, dim=-1, keepdim=True).values
            # n_min, n_max = torch.min(delta_n, dim=-1, keepdim=True).values, torch.max(delta_n, dim=-1, keepdim=True).values
            # p_thresh = p_min + torch.rand_like(p_thresh) * (p_max - p_min) * lerp
            # n_thresh = p_min + torch.rand_like(n_thresh) * (n_max - n_min) * lerp
            # -------------- #
        # estimate shared elements, then compute averaged vectors
        p_shared = (delta_p < p_thresh).detach()
        n_shared = (delta_n < n_thresh).detach()
        # done!
        return p_shared, n_shared

    @staticmethod
    def compute_ave(a_z_mean, p_z_mean, n_z_mean, lerp=None):
        # estimate shared elements, then compute averaged vectors
        p_shared, n_shared = AdaTripletVae.compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=lerp)
        # compute averaged
        ap_ave = (0.5 * a_z_mean) + (0.5 * p_z_mean)
        an_ave = (0.5 * a_z_mean) + (0.5 * n_z_mean)
        ap_a_ave, ap_p_ave = torch.where(p_shared, ap_ave, a_z_mean), torch.where(p_shared, ap_ave, p_z_mean)
        an_a_ave, an_n_ave = torch.where(n_shared, an_ave, a_z_mean), torch.where(n_shared, an_ave, n_z_mean)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        return ap_a_ave, ap_p_ave, an_a_ave, an_n_ave

    @staticmethod
    def compute_shared_loss(a_z_mean, p_z_mean, n_z_mean, lerp=None):
        p_shared_mask, n_shared_mask = AdaTripletVae.compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=lerp)
        p_shared_loss = torch.norm(torch.where(p_shared_mask, a_z_mean-p_z_mean, torch.zeros_like(a_z_mean)), p=2, dim=-1).mean()
        n_shared_loss = torch.norm(torch.where(n_shared_mask, a_z_mean-n_z_mean, torch.zeros_like(a_z_mean)), p=2, dim=-1).mean()
        shared_loss = 0.5 * p_shared_loss + 0.5 * n_shared_loss
        return shared_loss


def blend(a, b, alpha):
    """
    if alpha == 0 then a is returned
    if alpha == 1 then b is returned
    """
    alpha = np.clip(alpha, 0, 1)
    return ((1-alpha) * a) + (alpha * b)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
