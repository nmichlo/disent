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
from typing import Sequence

import torch
from torch.distributions import Normal

from disent.frameworks.helper.triplet_loss import configured_triplet, configured_dist_triplet
from disent.frameworks.vae.supervised import TripletVae
from disent.frameworks.vae.weaklysupervised import AdaVae
import logging


log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AdaTripletVae(TripletVae):

    REQUIRED_OBS = 3

    @dataclass
    class cfg(TripletVae.cfg, AdaVae.cfg):
        # adavae
        average_mode: str = None
        thresh_mode: str = None
        thresh_ratio: float = 0.5
        # adatvae
        ada_triplet_sample: bool = False
        ada_triplet_loss: str = 'triplet_soft_ave'
        ada_triplet_ratio: float = 1.0
        ada_triplet_soft_scale: float = 1.0

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # warn if unsupported variables are used
        if self.cfg.average_mode is not None: warnings.warn(f'{self.__class__.__name__} does not support {AdaVae.__name__}.cfg.average_mode')
        if self.cfg.thresh_mode is not None:  warnings.warn(f'{self.__class__.__name__} does not support {AdaVae.__name__}.cfg.thresh_mode')

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal], zs_sampled: Sequence[torch.Tensor], xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]):
        return self.compute_ada_triplet_loss(zs=self.get_representations(ds_posterior, cfg=self.cfg), cfg=self.cfg)

    @staticmethod
    def get_representations(ds_posterior: Sequence[Normal], cfg: cfg):
        if cfg.ada_triplet_sample:
            return [d.rsample() for d in ds_posterior]
        else:
            return [d.mean for d in ds_posterior]

    @staticmethod
    def compute_ada_triplet_loss(zs: Sequence[torch.Tensor], cfg: cfg):
        a_z, p_z, n_z = zs

        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # ADAPTIVE AVERAGING
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # perform averaging
        ap_shared_mask = AdaVae.compute_z_shared_mask(a_z, p_z, ratio=cfg.thresh_ratio)
        an_shared_mask = AdaVae.compute_z_shared_mask(a_z, n_z, ratio=cfg.thresh_ratio)
        pn_shared_mask = AdaVae.compute_z_shared_mask(p_z, n_z, ratio=cfg.thresh_ratio)

        # compute averaged
        ap_ave = (0.5 * a_z) + (0.5 * p_z)
        an_ave = (0.5 * a_z) + (0.5 * n_z)
        pn_ave = (0.5 * a_z) + (0.5 * n_z)
        ap_a_ave, ap_p_ave = torch.where(ap_shared_mask, ap_ave, a_z), torch.where(ap_shared_mask, ap_ave, p_z)
        an_a_ave, an_n_ave = torch.where(an_shared_mask, an_ave, a_z), torch.where(an_shared_mask, an_ave, n_z)
        pn_p_ave, pn_n_ave = torch.where(pn_shared_mask, pn_ave, p_z), torch.where(pn_shared_mask, pn_ave, n_z)

        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # Losses
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # Normal Triplet Loss
        trip_loss = configured_triplet(a_z, p_z, n_z, cfg=cfg)

        # Hard Averaging Before Triplet
        triplet_hard_ave     = configured_dist_triplet(pos_delta=ap_p_ave-ap_a_ave, neg_delta=an_n_ave-an_a_ave, cfg=cfg)
        triplet_hard_ave_neg = configured_dist_triplet(pos_delta=p_z-a_z,           neg_delta=an_n_ave-an_a_ave, cfg=cfg)

        # Hard All Averaging Before Triplet
        ave_a = (0.5 * ap_a_ave) + (0.5 * an_a_ave)
        ave_p = (0.5 * pn_p_ave) + (0.5 * ap_p_ave)
        ave_n = (0.5 * an_n_ave) + (0.5 * pn_n_ave)
        triplet_all_hard_ave = configured_dist_triplet(pos_delta=ave_a-ave_p, neg_delta=ave_a-ave_n, cfg=cfg)

        # MSE Averaging With Triplet Loss
        _soft_ave_ap_loss = torch.norm(torch.where(ap_shared_mask, a_z-p_z, torch.zeros_like(a_z)), p=2, dim=-1).mean()
        _soft_ave_an_loss = torch.norm(torch.where(an_shared_mask, a_z-n_z, torch.zeros_like(a_z)), p=2, dim=-1).mean()
        _soft_ave_pn_loss = torch.norm(torch.where(pn_shared_mask, p_z-n_z, torch.zeros_like(a_z)), p=2, dim=-1).mean()
        soft_ave_loss     = (_soft_ave_ap_loss + _soft_ave_an_loss) / 2
        soft_ave_neg_loss = (_soft_ave_an_loss)
        all_soft_ave_loss = (_soft_ave_ap_loss + _soft_ave_an_loss + _soft_ave_pn_loss) / 3

        losses = {
            'triplet':              trip_loss,
            # soft ave
            'triplet_soft_ave':     trip_loss + (cfg.ada_triplet_soft_scale * cfg.triplet_scale) * soft_ave_loss,
            'triplet_soft_neg_ave': trip_loss + (cfg.ada_triplet_soft_scale * cfg.triplet_scale) * soft_ave_neg_loss,
            'triplet_all_soft_ave': trip_loss + (cfg.ada_triplet_soft_scale * cfg.triplet_scale) * all_soft_ave_loss,
            # hard ave
            'triplet_hard_ave':     torch.lerp(trip_loss, triplet_hard_ave,     weight=cfg.ada_triplet_ratio),
            'triplet_hard_neg_ave': torch.lerp(trip_loss, triplet_hard_ave_neg, weight=cfg.ada_triplet_ratio),
            'triplet_all_hard_ave': torch.lerp(trip_loss, triplet_all_hard_ave, weight=cfg.ada_triplet_ratio),
        }

        return losses[cfg.ada_triplet_loss], {
            'triplet': trip_loss,
            'triplet_chosen': losses[cfg.ada_triplet_loss],
            'ap_shared': ap_shared_mask.sum(dim=1).float().mean(),
            'an_shared': an_shared_mask.sum(dim=1).float().mean(),
            'pn_shared': pn_shared_mask.sum(dim=1).float().mean(),
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
