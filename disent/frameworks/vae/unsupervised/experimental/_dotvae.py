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

from dataclasses import dataclass
from typing import Optional
from typing import Sequence

import logging

import kornia
import numpy as np
import torch
import torchvision
from torch.distributions import Normal

from disent.frameworks.vae.supervised.experimental._adatvae import AdaTripletVae
from experiment.util.hydra_utils import instantiate_recursive


log = logging.getLogger(__name__)


# ========================================================================= #
# tvae                                                                      #
# ========================================================================= #


class DataOverlapTripletVae(AdaTripletVae):

    REQUIRED_OBS = 1

    @dataclass
    class cfg(AdaTripletVae.cfg):
        # OVERRIDE - triplet vae configs
        detach: bool = False
        detach_decoder: bool = False
        detach_no_kl: bool = False
        detach_logvar: float = None  # std = 0.5, logvar = ln(std**2) ~= -2,77
        # OVERLAP VAE
        overlap_num: int = 1024
        overlap_mine_ratio: float = 0.1
        overlap_mine_triplet_mode: str = 'none'
        # AUGMENT
        overlap_augment_mode: str = 'none'
        overlap_augment: Optional[dict] = None

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # initialise
        if self.cfg.overlap_augment_mode != 'none':
            assert self.cfg.overlap_augment is not None, 'if cfg.overlap_augment_mode is not "none", then cfg.overlap_augment must be defined.'
        if self.cfg.overlap_augment is not None:
            self._augment = instantiate_recursive(self.cfg.overlap_augment)

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Normal], ds_prior, zs_sampled, xs_partial_recon, xs_targ: Sequence[torch.Tensor]):
        # get values
        (d_posterior,), (x_targ,) = ds_posterior, xs_targ
        # generate & mine random triples from batch -- this does not generate unique pairs
        a_idxs, p_idxs, n_idxs = torch.randint(len(x_targ), size=(3, min(self.cfg.overlap_num, len(x_targ)**3)))
        a_idxs, p_idxs, n_idxs = self.mine_triplets(x_targ, a_idxs, p_idxs, n_idxs)
        # make triples
        new_xs_targ = [x_targ[idxs] for idxs in (a_idxs, p_idxs, n_idxs)]
        new_ds_posterior = [Normal(d_posterior.loc[idxs], d_posterior.scale[idxs]) for idxs in (a_idxs, p_idxs, n_idxs)]
        # augment targets
        aug_xs_targ = self.augment_triplet_targets(new_xs_targ)
        # create new triplets
        ds_posterior_NEW = self.overlap_create_triplets(ds_posterior=new_ds_posterior, xs_targ=aug_xs_targ, cfg=self.cfg, unreduced_loss_fn=self.recon_handler.compute_unreduced_loss)
        # compute loss
        loss, logs = AdaTripletVae.estimate_ada_triplet_loss(
            zs_params=self.all_dist_to_params(ds_posterior_NEW),
            ds_posterior=ds_posterior_NEW,
            cfg=self.cfg,
        )
        return loss, {
            **logs,
            **DataOverlapTripletVae.overlap_measure_differences(new_ds_posterior, aug_xs_targ, unreduced_loss_fn=self.recon_handler.compute_unreduced_loss, unreduced_kl_loss_fn=self.latents_handler.compute_unreduced_kl_loss),
        }

    @staticmethod
    def overlap_create_triplets(ds_posterior, xs_targ, cfg: cfg, unreduced_loss_fn):
        # check the recon loss
        assert cfg.recon_loss == 'mse', 'only mse loss is supported'
        # CORE: order the latent variables for triplet
        swap_mask = DataOverlapTripletVae.overlap_swap_mask(xs_targ=xs_targ, unreduced_loss_fn=unreduced_loss_fn)
        ds_posterior_NEW = DataOverlapTripletVae.overlap_get_swapped(ds_posterior=ds_posterior, swap_mask=swap_mask)
        # return values
        return ds_posterior_NEW

    @staticmethod
    def overlap_swap_mask(xs_targ: Sequence[torch.Tensor], unreduced_loss_fn: callable) -> torch.Tensor:
        # get variables
        a_x_targ_OLD, p_x_targ_OLD, n_x_targ_OLD = xs_targ
        # CORE OF THIS APPROACH
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # calculate which are wrong!
        # TODO: add more loss functions, like perceptual & others
        a_p_losses = unreduced_loss_fn(a_x_targ_OLD, p_x_targ_OLD).mean(dim=(-3, -2, -1))  # (B, C, H, W) -> (B,)
        a_n_losses = unreduced_loss_fn(a_x_targ_OLD, n_x_targ_OLD).mean(dim=(-3, -2, -1))  # (B, C, H, W) -> (B,)
        swap_mask = (a_p_losses > a_n_losses)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        return swap_mask # (B,)

    @staticmethod
    def overlap_get_swapped(ds_posterior: Sequence[Normal], swap_mask: torch.Tensor) -> Sequence[Normal]:
        # get variables
        a_post_OLD, p_post_OLD, n_post_OLD = ds_posterior
        # check variables
        assert swap_mask.ndim == 1
        assert a_post_OLD.mean.ndim == 2
        assert len(swap_mask) == len(a_post_OLD.mean)
        # repeat mask along latent dim
        swap_mask = swap_mask[:, None].repeat(1, a_post_OLD.mean.shape[-1])  # (B,) -> (B, Z)
        # swap if wrong!
        p_z_loc   = torch.where(swap_mask, n_post_OLD.loc,   p_post_OLD.loc)
        n_z_loc   = torch.where(swap_mask, p_post_OLD.loc,   n_post_OLD.loc)
        p_z_scale = torch.where(swap_mask, n_post_OLD.scale, p_post_OLD.scale)
        n_z_scale = torch.where(swap_mask, p_post_OLD.scale, n_post_OLD.scale)
        # return new distributions
        return a_post_OLD, Normal(loc=p_z_loc, scale=p_z_scale), Normal(loc=n_z_loc, scale=n_z_scale)

    def augment_triplet_targets(self, xs_targ):
        if self.cfg.overlap_augment_mode == 'none':
            return xs_targ
        elif (self.cfg.overlap_augment_mode == 'augment') or (self.cfg.overlap_augment_mode == 'augment_each'):
            # recreate augment each time
            if self.cfg.overlap_augment_mode == 'augment_each':
                self._augment = instantiate_recursive(self.cfg.augments)
            # augment on correct device
            aug_xs_targ = [self._augment(x_targ) for x_targ in xs_targ]
            # checks
            assert all(a.shape == b.shape for a, b in zip(xs_targ, aug_xs_targ))
            return aug_xs_targ
        else:
            raise KeyError(f'invalid cfg.overlap_augment_mode={repr(self.cfg.overlap_augment_mode)}')

    def mine_triplets(self, x_targ, a_idxs, p_idxs, n_idxs):
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # CUSTOM MODE
        overlap_mine_triplet_mode = self.cfg.overlap_mine_triplet_mode
        if overlap_mine_triplet_mode.startswith('random_'):
            overlap_mine_triplet_mode = np.random.choice(overlap_mine_triplet_mode[len('random_'):].split('_or_'))
        # TRIPLET MINING - TODO: this can be moved into separate functions
        # - Improved Embeddings with Easy Positive Triplet Mining (1904.04370v2)
        # - https://stats.stackexchange.com/questions/475655
        if overlap_mine_triplet_mode == 'semi_hard_neg':
            # SEMI HARD NEGATIVE MINING
            # "choose an anchor-negative pair that is farther than the anchor-positive pair, but within the margin, and so still contributes a positive loss"
            # -- triples satisfy d(a, p) < d(a, n) < alpha
            d_a_p = self.recon_handler.compute_unreduced_loss(x_targ[a_idxs], x_targ[p_idxs]).mean(dim=(-3, -2, -1))
            d_a_n = self.recon_handler.compute_unreduced_loss(x_targ[a_idxs], x_targ[n_idxs]).mean(dim=(-3, -2, -1))
            alpha = self.cfg.triplet_margin_max
            # get hard negatives
            semi_hard_mask = (d_a_p < d_a_n) & (d_a_n < alpha)
            # get indices
            if torch.sum(semi_hard_mask) > 0:
                a_idxs, p_idxs, n_idxs = a_idxs[semi_hard_mask], p_idxs[semi_hard_mask], n_idxs[semi_hard_mask]
            else:
                log.warning('no semi_hard negatives found! using entire batch')
        elif overlap_mine_triplet_mode == 'hard_neg':
            # HARD NEGATIVE MINING
            # "most similar images which have a different label from the anchor image"
            # -- triples with smallest d(a, n)
            d_a_n = self.recon_handler.compute_unreduced_loss(x_targ[a_idxs], x_targ[n_idxs]).mean(dim=(-3, -2, -1))
            # get hard negatives
            hard_idxs = torch.argsort(d_a_n, descending=False)[:int(self.cfg.overlap_num * self.cfg.overlap_mine_ratio)]
            # get indices
            a_idxs, p_idxs, n_idxs = a_idxs[hard_idxs], p_idxs[hard_idxs], n_idxs[hard_idxs]
        elif overlap_mine_triplet_mode == 'easy_neg':
            # EASY NEGATIVE MINING
            # "least similar images which have the different label from the anchor image"
            raise RuntimeError('This triplet mode is not useful! Choose another.')
        elif overlap_mine_triplet_mode == 'hard_pos':
            # HARD POSITIVE MINING -- this performs really well!
            # "least similar images which have the same label to as anchor image"
            # -- shown not to be suitable for all datasets
            d_a_p = self.recon_handler.compute_unreduced_loss(x_targ[a_idxs], x_targ[p_idxs]).mean(dim=(-3, -2, -1))
            # get hard positives
            hard_idxs = torch.argsort(d_a_p, descending=True)[:int(self.cfg.overlap_num * self.cfg.overlap_mine_ratio)]
            # get indices
            a_idxs, p_idxs, n_idxs = a_idxs[hard_idxs], p_idxs[hard_idxs], n_idxs[hard_idxs]
        elif overlap_mine_triplet_mode == 'easy_pos':
            # EASY POSITIVE MINING
            # "the most similar images that have the same label as the anchor image"
            d_a_p = self.recon_handler.compute_unreduced_loss(x_targ[a_idxs], x_targ[p_idxs]).mean(dim=(-3, -2, -1))
            # get easy positives
            easy_idxs = torch.argsort(d_a_p, descending=False)[:int(self.cfg.overlap_num * self.cfg.overlap_mine_ratio)]
            # get indices
            a_idxs, p_idxs, n_idxs = a_idxs[easy_idxs], p_idxs[easy_idxs], n_idxs[easy_idxs]
        elif overlap_mine_triplet_mode != 'none':
            raise KeyError
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        return a_idxs, p_idxs, n_idxs

    @staticmethod
    def overlap_measure_differences(ds_posterior, xs_targ, unreduced_loss_fn, unreduced_kl_loss_fn):
        d0_posterior, d1_posterior, d2_posterior = ds_posterior
        x0_targ, x1_targ, x2_targ = xs_targ

        # TESTS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # EFFECTIVELY SUMMED MSE WITH TERM: (u_1 - u_0)**2
        kl_0_1 = unreduced_kl_loss_fn(d0_posterior, d1_posterior)
        kl_0_2 = unreduced_kl_loss_fn(d0_posterior, d2_posterior)
        mu_0_1 = (d0_posterior.mean - d1_posterior.mean) ** 2
        mu_0_2 = (d0_posterior.mean - d2_posterior.mean) ** 2
        # z differences
        kl_mu_differences_all = torch.mean(((kl_0_1 < kl_0_2) ^ (mu_0_1 < mu_0_2)).to(torch.float32))
        kl_mu_differences_ave = torch.mean(((kl_0_1.mean(dim=-1) < kl_0_2.mean(dim=-1)) ^ (mu_0_1.mean(dim=-1) < mu_0_2.mean(dim=-1))).to(torch.float32))
        # obs differences
        xs_0_1 = unreduced_loss_fn(x0_targ, x1_targ).mean(dim=(-3, -2, -1))
        xs_0_2 = unreduced_loss_fn(x0_targ, x2_targ).mean(dim=(-3, -2, -1))
        # get differences
        kl_xs_differences_all = torch.mean(((kl_0_1 < kl_0_2) ^ (xs_0_1 < xs_0_2)[..., None]).to(torch.float32))
        mu_xs_differences_all = torch.mean(((mu_0_1 < mu_0_2) ^ (xs_0_1 < xs_0_2)[..., None]).to(torch.float32))
        kl_xs_differences_ave = torch.mean(((kl_0_1.mean(dim=-1) < kl_0_2.mean(dim=-1)) ^ (xs_0_1 < xs_0_2)).to(torch.float32))
        mu_xs_differences_ave = torch.mean(((mu_0_1.mean(dim=-1) < mu_0_2.mean(dim=-1)) ^ (xs_0_1 < xs_0_2)).to(torch.float32))
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        return {
            'overlap_kl_mu_diffs_all': kl_mu_differences_all,
            'overlap_kl_mu_diffs_ave': kl_mu_differences_ave,
            'overlap_kl_xs_diffs_all': kl_xs_differences_all,
            'overlap_mu_xs_diffs_all': mu_xs_differences_all,
            'overlap_kl_xs_diffs_ave': kl_xs_differences_ave,
            'overlap_mu_xs_diffs_ave': mu_xs_differences_ave,
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
