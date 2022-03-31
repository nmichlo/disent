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
from typing import Tuple

import torch
from torch.distributions import Distribution
from torch.distributions import Normal

from disent.nn.loss.triplet import configured_dist_triplet
from disent.nn.loss.triplet import configured_triplet
from disent.frameworks.vae._supervised__tvae import TripletVae
from disent.frameworks.vae._weaklysupervised__adavae import AdaVae


log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AdaNegTripletVae(TripletVae):
    """
    Michlo et al.
    https://github.com/nmichlo/msc-research

    This is the supervised version of the Adaptive Triplet Loss for Variational Auto-Encoders
    - Triplets are usually ordered using the ground-truth distances over observations
    - Adaptive Triplet Loss is used to guide distance learning and encourage disentanglement
    """

    REQUIRED_OBS = 3

    @dataclass
    class cfg(TripletVae.cfg, AdaVae.cfg):
        # adavae
        ada_thresh_mode: str = 'dist'  # only works for: adat_share_mask_mode == "posterior"
        # ada_tvae - loss
        # * this should be used with a schedule, slowly decrease from 1.0 down to 0.5 or less
        # * a similar schedule should also be used on `ada_thresh_ratio`, slowly increasing from 0.0 to 0.5
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
# AveAda-TVAE                                                               #
# ========================================================================= #


@dataclass
class AdaTripletVae_cfg(TripletVae.cfg, AdaVae.cfg):
    # adavae
    ada_thresh_mode: str = 'dist'  # only works for: adat_share_mask_mode == "posterior"
    # ada_tvae - averaging
    adat_share_mask_mode: str = 'posterior'


def compute_triplet_shared_masks_from_zs(zs: Sequence[torch.Tensor], cfg):
    """
    required config params:
    - cfg.ada_thresh_ratio:
    """
    a_z, p_z, n_z = zs
    # shared elements that need to be averaged, computed per pair in the batch.
    ap_share_mask = AdaVae.compute_shared_mask_from_zs(a_z, p_z, ratio=cfg.ada_thresh_ratio)
    an_share_mask = AdaVae.compute_shared_mask_from_zs(a_z, n_z, ratio=cfg.ada_thresh_ratio)
    pn_share_mask = AdaVae.compute_shared_mask_from_zs(p_z, n_z, ratio=cfg.ada_thresh_ratio)
    # return values
    share_masks = (ap_share_mask, an_share_mask, pn_share_mask)
    return share_masks, {
        'ap_shared': ap_share_mask.sum(dim=1).float().mean(),
        'an_shared': an_share_mask.sum(dim=1).float().mean(),
        'pn_shared': pn_share_mask.sum(dim=1).float().mean(),
    }


def compute_triplet_shared_masks(ds_posterior: Sequence[Distribution], cfg: AdaTripletVae_cfg):
    """
    required config params:
    - cfg.ada_thresh_ratio:
    - cfg.ada_thresh_mode: "kl", "symmetric_kl", "dist", "sampled_dist"
      : only applies if cfg.ada_share_mask_mode=="posterior"
    - cfg.adat_share_mask_mode: "posterior", "sample", "sample_each"
    """
    a_posterior, p_posterior, n_posterior = ds_posterior

    # shared elements that need to be averaged, computed per pair in the batch.
    if cfg.adat_share_mask_mode == 'posterior':
        ap_share_mask = AdaVae.compute_shared_mask_from_posteriors(a_posterior, p_posterior, thresh_mode=cfg.ada_thresh_mode, ratio=cfg.ada_thresh_ratio)
        an_share_mask = AdaVae.compute_shared_mask_from_posteriors(a_posterior, n_posterior, thresh_mode=cfg.ada_thresh_mode, ratio=cfg.ada_thresh_ratio)
        pn_share_mask = AdaVae.compute_shared_mask_from_posteriors(p_posterior, n_posterior, thresh_mode=cfg.ada_thresh_mode, ratio=cfg.ada_thresh_ratio)
    elif cfg.adat_share_mask_mode == 'sample':
        a_z_sample, p_z_sample, n_z_sample = a_posterior.rsample(), p_posterior.rsample(), n_posterior.rsample()
        ap_share_mask = AdaVae.compute_shared_mask_from_zs(a_z_sample, p_z_sample, ratio=cfg.ada_thresh_ratio)
        an_share_mask = AdaVae.compute_shared_mask_from_zs(a_z_sample, n_z_sample, ratio=cfg.ada_thresh_ratio)
        pn_share_mask = AdaVae.compute_shared_mask_from_zs(p_z_sample, n_z_sample, ratio=cfg.ada_thresh_ratio)
    elif cfg.adat_share_mask_mode == 'sample_each':
        ap_share_mask = AdaVae.compute_shared_mask_from_zs(a_posterior.rsample(), p_posterior.rsample(), ratio=cfg.ada_thresh_ratio)
        an_share_mask = AdaVae.compute_shared_mask_from_zs(a_posterior.rsample(), n_posterior.rsample(), ratio=cfg.ada_thresh_ratio)
        pn_share_mask = AdaVae.compute_shared_mask_from_zs(p_posterior.rsample(), n_posterior.rsample(), ratio=cfg.ada_thresh_ratio)
    else:
        raise KeyError(f'Invalid cfg.adat_share_mask_mode={repr(cfg.adat_share_mask_mode)}')

    # return values
    share_masks = (ap_share_mask, an_share_mask, pn_share_mask)
    return share_masks, {
        'ap_shared': ap_share_mask.sum(dim=1).float().mean(),
        'an_shared': an_share_mask.sum(dim=1).float().mean(),
        'pn_shared': pn_share_mask.sum(dim=1).float().mean(),
    }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
