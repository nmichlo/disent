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

from disent.frameworks.helper.triplet_loss import configured_dist_triplet
from disent.frameworks.helper.triplet_loss import configured_triplet
from disent.frameworks.vae._supervised__tvae import TripletVae
from disent.frameworks.vae.experimental._supervised__adatvae import compute_triplet_shared_masks
from disent.frameworks.vae.experimental._supervised__adatvae import compute_triplet_shared_masks_from_zs
from disent.frameworks.vae._weaklysupervised__adavae import AdaVae


log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AdaNegTripletVae(TripletVae):

    """
    This is a condensed version of the ada_tvae and adaave_tvae,
    using approximately the best settings and loss...
    """

    REQUIRED_OBS = 3

    @dataclass
    class cfg(TripletVae.cfg, AdaVae.cfg):
        # adavae
        ada_thresh_mode: str = 'dist'  # only works for: adat_share_mask_mode == "posterior"
        # ada_tvae - loss
        adat_triplet_share_scale: float = 0.95
        # ada_tvae - averaging
        adat_share_mask_mode: str = 'posterior'

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Normal], ds_prior: Sequence[Normal], zs_sampled: Sequence[torch.Tensor], xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]):
        return self.estimate_ada_triplet_loss(
            ds_posterior=ds_posterior,
            cfg=self.cfg,
        )

    @staticmethod
    def estimate_ada_triplet_loss_from_zs(zs: Sequence[torch.Tensor], cfg: cfg):
        # compute shared masks, shared embeddings & averages over shared embeddings
        share_masks, share_logs = compute_triplet_shared_masks_from_zs(zs=zs, cfg=cfg)
        # compute loss
        ada_triplet_loss, ada_triplet_logs = AdaNegTripletVae.compute_ada_triplet_loss(share_masks=share_masks, zs=zs, cfg=cfg)
        # merge logs & return loss
        return ada_triplet_loss, {
            **ada_triplet_logs,
            **share_logs,
        }

    @staticmethod
    def estimate_ada_triplet_loss(ds_posterior: Sequence[Normal], cfg: cfg):
        # compute shared masks, shared embeddings & averages over shared embeddings
        share_masks, share_logs = compute_triplet_shared_masks(ds_posterior, cfg=cfg)
        # compute loss
        ada_triplet_loss, ada_triplet_logs = AdaNegTripletVae.compute_ada_triplet_loss(share_masks=share_masks, zs=(d.mean for d in ds_posterior), cfg=cfg)
        # merge logs & return loss
        return ada_triplet_loss, {
            **ada_triplet_logs,
            **share_logs,
        }

    @staticmethod
    def compute_ada_triplet_loss(share_masks, zs, cfg: cfg):
        # Normal Triplet Loss
        (a_z, p_z, n_z) = zs
        trip_loss = configured_triplet(a_z, p_z, n_z, cfg=cfg)

        # Soft Scaled Negative Triplet
        (ap_share_mask, an_share_mask, pn_share_mask) = share_masks
        triplet_hard_neg_ave_scaled = configured_dist_triplet(
            pos_delta=a_z - p_z,
            neg_delta=torch.where(an_share_mask, cfg.adat_triplet_share_scale * (a_z - n_z), (a_z - n_z)),
            cfg=cfg,
        )

        return triplet_hard_neg_ave_scaled, {
            'triplet': trip_loss,
            'triplet_chosen': triplet_hard_neg_ave_scaled,
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
