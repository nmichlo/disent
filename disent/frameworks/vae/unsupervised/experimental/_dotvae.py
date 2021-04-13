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

import numpy as np
import torch
from torch.distributions import Normal

from disent.frameworks.vae.supervised import AdaNegTripletVae
from experiment.util.hydra_utils import instantiate_recursive


log = logging.getLogger(__name__)


# ========================================================================= #
# tvae                                                                      #
# ========================================================================= #


class DataOverlapTripletVae(AdaNegTripletVae):

    REQUIRED_OBS = 1

    @dataclass
    class cfg(AdaNegTripletVae.cfg):
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
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # 1. augment batch
        with torch.no_grad():
            xs_targ = self.augment_triplet_targets(xs_targ)
        (d_posterior,), (x_targ,) = ds_posterior, xs_targ
        # 2. generate random triples -- this does not generate unique pairs
        a_idxs, p_idxs, n_idxs = torch.randint(len(x_targ), size=(3, min(self.cfg.overlap_num, len(x_targ)**3)), device=x_targ.device)
        # 3. reorder random triples
        a_idxs, p_idxs, n_idxs = self.overlap_swap_triplet_idxs(x_targ, a_idxs, p_idxs, n_idxs, cfg=self.cfg, unreduced_loss_fn=self.recon_handler.compute_unreduced_loss)
        # 4. mine random triples
        a_idxs, p_idxs, n_idxs = self.mine_triplets(x_targ, a_idxs, p_idxs, n_idxs)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # 5. compute triplet loss
        return AdaNegTripletVae.estimate_ada_triplet_loss(
            ds_posterior=[Normal(d_posterior.loc[idxs], d_posterior.scale[idxs]) for idxs in (a_idxs, p_idxs, n_idxs)],
            cfg=self.cfg,
        )

    @staticmethod
    def overlap_swap_triplet_idxs(x_targ, a_idxs, p_idxs, n_idxs, cfg: cfg, unreduced_loss_fn):
        xs_targ = [x_targ[idxs] for idxs in (a_idxs, p_idxs, n_idxs)]
        # check the recon loss
        assert cfg.recon_loss == 'mse', 'only mse loss is supported'
        # CORE: order the latent variables for triplet
        swap_mask = DataOverlapTripletVae.overlap_swap_mask(xs_targ=xs_targ, unreduced_loss_fn=unreduced_loss_fn)
        # swap all idxs
        swapped_a_idxs = a_idxs
        swapped_p_idxs = torch.where(swap_mask, n_idxs, p_idxs)
        swapped_n_idxs = torch.where(swap_mask, p_idxs, n_idxs)
        # return values
        return swapped_a_idxs, swapped_p_idxs, swapped_n_idxs


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
            raise KeyError(f'invalid cfg.overlap_mine_triplet_mode=={repr(self.cfg.overlap_mine_triplet_mode)}')
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        return a_idxs, p_idxs, n_idxs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
