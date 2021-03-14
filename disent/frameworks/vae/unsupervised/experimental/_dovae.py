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
from typing import Sequence

import logging

import numpy as np
import torch
from torch.distributions import Normal

from disent.frameworks.vae.supervised.experimental._adatvae import AdaTripletVae


log = logging.getLogger(__name__)


# ========================================================================= #
# tvae                                                                      #
# ========================================================================= #


class DataOverlapVae(AdaTripletVae):

    REQUIRED_OBS = 1

    @dataclass
    class cfg(AdaTripletVae.cfg):
        # OVERRIDE - triplet vae configs
        detach: bool = True
        detach_decoder: bool = True
        detach_no_kl: bool = False
        detach_logvar: float = 0  # std = 0.5, logvar = ln(std**2) ~= -2,77
        # OVERRIDE - triplet loss configs
        triplet_scale: float = 0
        # OVERRIDE ADAVAE
        # adatvae: what version of triplet to use
        triplet_mode: str = 'lerp_trip_to_mse_ada'  # 'ada_p_orig_lerp'
        # adatvae: annealing
        lerp_step_start: int = 3600
        lerp_step_end: int = 14400
        lerp_goal: float = 0.25
        # OVERLAP VAE
        overlap_triplet_mode: str = 'triplet'
        overlap_num: int = 1024
        overlap_z_mode: str = 'mean'
        # TRIPLET MINING
        overlap_mine_triplet_mode: str = 'none'
        overlap_mine_ratio: float = 0.1

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Normal], ds_prior, zs_sampled, xs_partial_recon, xs_targ: Sequence[torch.Tensor]):
        # get values
        (d_posterior,), (x_targ,) = ds_posterior, xs_targ
        # adavae
        self.step += 1

        # generate random triples -- TODO: this does not generate unique pairs
        a_idxs, p_idxs, n_idxs = torch.randint(len(x_targ), size=(3, min(self.cfg.overlap_num, len(x_targ)**3)))
        # mine triplets
        a_idxs, p_idxs, n_idxs = self.mine_triplets(x_targ, a_idxs, p_idxs, n_idxs)

        # make triples
        xs_targ_NEW = [x_targ[idxs] for idxs in (a_idxs, p_idxs, n_idxs)]
        ds_posterior_NEW = [Normal(d_posterior.loc[idxs], d_posterior.scale[idxs]) for idxs in (a_idxs, p_idxs, n_idxs)]

        # make representation triples
        if self.cfg.overlap_z_mode == 'means':
            zs = [d.mean for d in ds_posterior_NEW]
        elif self.cfg.overlap_z_mode == 'samples':
            zs = [d.rsample() for d in ds_posterior_NEW]  # we resample here in case z_sampled is detached
        else:
            raise KeyError(f'invalid cfg.overlap_z_mode: {repr(self.cfg.overlap_z_mode)}')

        # compute loss
        loss, logs = self.compute_overlap_triplet_loss(zs_mean=zs, xs_targ=xs_targ_NEW, cfg=self.cfg, step=self.step, unreduced_loss_fn=self.recon_handler.compute_unreduced_loss)

        return loss, {
            **logs,
            **DataOverlapVae.overlap_measure_differences(ds_posterior_NEW, xs_targ_NEW, unreduced_loss_fn=self.recon_handler.compute_unreduced_loss, unreduced_kl_loss_fn=self.latents_handler.compute_unreduced_kl_loss),
        }

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
    def compute_overlap_triplet_loss(zs_mean, xs_targ, cfg, step: int, unreduced_loss_fn):
        # check the recon loss
        assert cfg.recon_loss == 'mse', 'only mse loss is supported'

        # CORE: order the latent variables for triplet
        zs_mean = DataOverlapVae.overlap_swap_zs(zs_mean=zs_mean, xs_targ=xs_targ, unreduced_loss_fn=unreduced_loss_fn)

        # compute the triplet loss
        if cfg.overlap_triplet_mode == 'triplet':
            triplet_loss, logs_triplet = AdaTripletVae.compute_triplet_loss(zs_mean=zs_mean, cfg=cfg)
        elif cfg.overlap_triplet_mode == 'ada_triplet':
            triplet_loss, logs_triplet = AdaTripletVae.compute_ada_triplet_loss(zs_mean=zs_mean, step=step, cfg=cfg)
        else:  # pragma: no cover
            raise KeyError

        return triplet_loss, {
            'overlap_triplet_loss': triplet_loss,
            **logs_triplet,
        }

    @staticmethod
    def overlap_swap_zs(zs_mean, xs_targ, unreduced_loss_fn):
        # get variables
        a_z_mean_OLD, p_z_mean_OLD, n_z_mean_OLD = zs_mean
        a_x_targ_OLD, p_x_targ_OLD, n_x_targ_OLD = xs_targ

        # CORE OF THIS APPROACH
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # calculate which are wrong!
        a_p_losses = unreduced_loss_fn(a_x_targ_OLD, p_x_targ_OLD).mean(dim=(-3, -2, -1))
        a_n_losses = unreduced_loss_fn(a_x_targ_OLD, n_x_targ_OLD).mean(dim=(-3, -2, -1))
        # swap if wrong!
        swap_mask = (a_p_losses > a_n_losses)[:, None].repeat(1, p_z_mean_OLD.shape[-1])
        p_z_mean = torch.where(swap_mask, n_z_mean_OLD, p_z_mean_OLD)
        n_z_mean = torch.where(swap_mask, p_z_mean_OLD, n_z_mean_OLD)
        # ++++++++++++++++++++++++++++++++++++++++++ #

        return a_z_mean_OLD, p_z_mean, n_z_mean

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
