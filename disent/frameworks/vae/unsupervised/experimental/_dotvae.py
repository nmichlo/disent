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
from typing import final
from typing import Optional
from typing import Sequence

import logging

import numpy as np
import torch
from torch.distributions import Normal

from disent.frameworks.helper.reconstructions import make_reconstruction_loss
from disent.frameworks.helper.reconstructions import ReconLossHandler
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
        overlap_loss: Optional[str] = None  # if None, use the value from recon_loss
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
        # get overlap loss
        overlap_loss = self.cfg.overlap_loss if (self.cfg.overlap_loss is not None) else self.cfg.recon_loss
        self.__overlap_handler: ReconLossHandler = make_reconstruction_loss(overlap_loss, reduction='mean')

    @final
    @property
    def overlap_handler(self) -> ReconLossHandler:
        return self.__overlap_handler

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Normal], ds_prior, zs_sampled, xs_partial_recon, xs_targ: Sequence[torch.Tensor]):
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # 1. augment batch
        (x_targ_orig,) = xs_targ
        with torch.no_grad():
            xs_targ = self.augment_triplet_targets(xs_targ)
        (d_posterior,), (x_targ,) = ds_posterior, xs_targ
        # 2. generate random triples -- this does not generate unique pairs
        a_idxs, p_idxs, n_idxs = torch.randint(len(x_targ), size=(3, min(self.cfg.overlap_num, len(x_targ)**3)), device=x_targ.device)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # self.debug(x_targ_orig, x_targ, a_idxs, p_idxs, n_idxs)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # 3. reorder random triples
        a_idxs, p_idxs, n_idxs = self.overlap_swap_triplet_idxs(x_targ, a_idxs, p_idxs, n_idxs)
        # 4. mine random triples
        a_idxs, p_idxs, n_idxs = self.mine_triplets(x_targ, a_idxs, p_idxs, n_idxs)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # 5. compute triplet loss
        loss, loss_log = AdaNegTripletVae.estimate_ada_triplet_loss(
            ds_posterior=[Normal(d_posterior.loc[idxs], d_posterior.scale[idxs]) for idxs in (a_idxs, p_idxs, n_idxs)],
            cfg=self.cfg,
        )
        return loss, {
            **loss_log,
        }

    # def debug(self, x_targ_orig, x_targ, a_idxs, p_idxs, n_idxs):
    #     a_p_overlap_orig = - self.recon_handler.compute_unreduced_loss(x_targ_orig[a_idxs], x_targ_orig[p_idxs]).mean(dim=(-3, -2, -1))  # (B, C, H, W) -> (B,)
    #     a_n_overlap_orig = - self.recon_handler.compute_unreduced_loss(x_targ_orig[a_idxs], x_targ_orig[n_idxs]).mean(dim=(-3, -2, -1))  # (B, C, H, W) -> (B,)
    #     a_p_overlap = - self.recon_handler.compute_unreduced_loss(x_targ[a_idxs], x_targ[p_idxs]).mean(dim=(-3, -2, -1))  # (B, C, H, W) -> (B,)
    #     a_n_overlap = - self.recon_handler.compute_unreduced_loss(x_targ[a_idxs], x_targ[n_idxs]).mean(dim=(-3, -2, -1))  # (B, C, H, W) -> (B,)
    #     a_p_overlap_mul = - (a_p_overlap_orig * a_p_overlap)
    #     a_n_overlap_mul = - (a_n_overlap_orig * a_n_overlap)
    #     # check number of things
    #     (up_values_orig, up_counts_orig) = torch.unique(a_p_overlap_orig, sorted=True, return_inverse=False, return_counts=True)
    #     (un_values_orig, un_counts_orig) = torch.unique(a_n_overlap_orig, sorted=True, return_inverse=False, return_counts=True)
    #     (up_values, up_counts) = torch.unique(a_p_overlap, sorted=True, return_inverse=False, return_counts=True)
    #     (un_values, un_counts) = torch.unique(a_n_overlap, sorted=True, return_inverse=False, return_counts=True)
    #     (up_values_mul, up_counts_mul) = torch.unique(a_p_overlap_mul, sorted=True, return_inverse=False, return_counts=True)
    #     (un_values_mul, un_counts_mul) = torch.unique(a_n_overlap_mul, sorted=True, return_inverse=False, return_counts=True)
    #     # plot!
    #     plt.scatter(up_values_orig.detach().cpu(), torch.cumsum(up_counts_orig, dim=-1).detach().cpu())
    #     plt.scatter(un_values_orig.detach().cpu(), torch.cumsum(un_counts_orig, dim=-1).detach().cpu())
    #     plt.scatter(up_values.detach().cpu(), torch.cumsum(up_counts, dim=-1).detach().cpu())
    #     plt.scatter(un_values.detach().cpu(), torch.cumsum(un_counts, dim=-1).detach().cpu())
    #     plt.scatter(up_values_mul.detach().cpu(), torch.cumsum(up_counts_mul, dim=-1).detach().cpu())
    #     plt.scatter(un_values_mul.detach().cpu(), torch.cumsum(un_counts_mul, dim=-1).detach().cpu())
    #     plt.show()
    #     time.sleep(10)

    def overlap_swap_triplet_idxs(self, x_targ, a_idxs, p_idxs, n_idxs):
        xs_targ = [x_targ[idxs] for idxs in (a_idxs, p_idxs, n_idxs)]
        # CORE: order the latent variables for triplet
        swap_mask = self.overlap_swap_mask(xs_targ=xs_targ)
        # swap all idxs
        swapped_a_idxs = a_idxs
        swapped_p_idxs = torch.where(swap_mask, n_idxs, p_idxs)
        swapped_n_idxs = torch.where(swap_mask, p_idxs, n_idxs)
        # return values
        return swapped_a_idxs, swapped_p_idxs, swapped_n_idxs

    def overlap_swap_mask(self, xs_targ: Sequence[torch.Tensor]) -> torch.Tensor:
        # get variables
        a_x_targ_OLD, p_x_targ_OLD, n_x_targ_OLD = xs_targ
        # CORE OF THIS APPROACH
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # calculate which are wrong!
        # TODO: add more loss functions, like perceptual & others
        with torch.no_grad():
            a_p_losses = self.overlap_handler.compute_pairwise_loss(a_x_targ_OLD, p_x_targ_OLD)  # (B, C, H, W) -> (B,)
            a_n_losses = self.overlap_handler.compute_pairwise_loss(a_x_targ_OLD, n_x_targ_OLD)  # (B, C, H, W) -> (B,)
            swap_mask = (a_p_losses > a_n_losses)  # (B,)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        return swap_mask

    def augment_triplet_targets(self, xs_targ):
        if self.cfg.overlap_augment_mode == 'none':
            aug_xs_targ = xs_targ
        elif (self.cfg.overlap_augment_mode == 'augment') or (self.cfg.overlap_augment_mode == 'augment_each'):
            # recreate augment each time
            if self.cfg.overlap_augment_mode == 'augment_each':
                self._augment = instantiate_recursive(self.cfg.augments)
            # augment on correct device
            aug_xs_targ = [self._augment(x_targ) for x_targ in xs_targ]
            # checks
            assert all(a.shape == b.shape for a, b in zip(xs_targ, aug_xs_targ))
        else:
            raise KeyError(f'invalid cfg.overlap_augment_mode={repr(self.cfg.overlap_augment_mode)}')
        return aug_xs_targ

    def mine_triplets(self, x_targ, a_idxs, p_idxs, n_idxs):
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # CUSTOM MODE
        overlap_mine_triplet_mode = self.cfg.overlap_mine_triplet_mode
        if overlap_mine_triplet_mode.startswith('random_'):
            overlap_mine_triplet_mode = np.random.choice(overlap_mine_triplet_mode[len('random_'):].split('_or_'))
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # get mining function
        try:
            mine_fn = _TRIPLET_MINE_MODES[overlap_mine_triplet_mode]
        except KeyError:
            raise KeyError(f'invalid cfg.overlap_mine_triplet_mode=={repr(self.cfg.overlap_mine_triplet_mode)}')
        # mine triplets -- can return array of indices or boolean mask array
        return mine_fn(
            x_targ=x_targ,
            a_idxs=a_idxs,
            p_idxs=p_idxs,
            n_idxs=n_idxs,
            cfg=self.cfg,
            pairwise_loss_fn=self.overlap_handler.compute_pairwise_loss
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


def mine_none(x_targ, a_idxs, p_idxs, n_idxs, cfg, pairwise_loss_fn):
    return a_idxs, p_idxs, n_idxs


def mine_semi_hard_neg(x_targ, a_idxs, p_idxs, n_idxs, cfg, pairwise_loss_fn):
    # SEMI HARD NEGATIVE MINING
    # "choose an anchor-negative pair that is farther than the anchor-positive pair, but within the margin, and so still contributes a positive loss"
    # -- triples satisfy d(a, p) < d(a, n) < alpha
    with torch.no_grad():
        d_a_p = pairwise_loss_fn(x_targ[a_idxs], x_targ[p_idxs])
        d_a_n = pairwise_loss_fn(x_targ[a_idxs], x_targ[n_idxs])
    # get hard negatives
    semi_hard_mask = (d_a_p < d_a_n) & (d_a_n < cfg.triplet_margin_max)
    # get indices
    if torch.sum(semi_hard_mask) > 0:
        return a_idxs[semi_hard_mask], p_idxs[semi_hard_mask], n_idxs[semi_hard_mask]
    else:
        log.warning('no semi_hard negatives found! using entire batch')
        return a_idxs, p_idxs, n_idxs


def mine_hard_neg(x_targ, a_idxs, p_idxs, n_idxs, cfg, pairwise_loss_fn):
    # HARD NEGATIVE MINING
    # "most similar images which have a different label from the anchor image"
    # -- triples with smallest d(a, n)
    with torch.no_grad():
        d_a_n = pairwise_loss_fn(x_targ[a_idxs], x_targ[n_idxs])
    # get hard negatives
    hard_idxs = torch.argsort(d_a_n, descending=False)[:int(cfg.overlap_num * cfg.overlap_mine_ratio)]
    # get indices
    return a_idxs[hard_idxs], p_idxs[hard_idxs], n_idxs[hard_idxs]


def mine_easy_neg(x_targ, a_idxs, p_idxs, n_idxs, cfg, pairwise_loss_fn):
    # EASY NEGATIVE MINING
    # "least similar images which have the different label from the anchor image"
    raise RuntimeError('This triplet mode is not useful! Choose another.')


def mine_hard_pos(x_targ, a_idxs, p_idxs, n_idxs, cfg, pairwise_loss_fn):
    # HARD POSITIVE MINING -- this performs really well!
    # "least similar images which have the same label to as anchor image"
    # -- shown not to be suitable for all datasets
    with torch.no_grad():
        d_a_p = pairwise_loss_fn(x_targ[a_idxs], x_targ[p_idxs])
    # get hard positives
    hard_idxs = torch.argsort(d_a_p, descending=True)[:int(cfg.overlap_num * cfg.overlap_mine_ratio)]
    # get indices
    return a_idxs[hard_idxs], p_idxs[hard_idxs], n_idxs[hard_idxs]


def mine_easy_pos(x_targ, a_idxs, p_idxs, n_idxs, cfg, pairwise_loss_fn):
    # EASY POSITIVE MINING
    # "the most similar images that have the same label as the anchor image"
    with torch.no_grad():
        d_a_p = pairwise_loss_fn(x_targ[a_idxs], x_targ[p_idxs])
    # get easy positives
    easy_idxs = torch.argsort(d_a_p, descending=False)[:int(cfg.overlap_num * cfg.overlap_mine_ratio)]
    # get indices
    return a_idxs[easy_idxs], p_idxs[easy_idxs], n_idxs[easy_idxs]


_TRIPLET_MINE_MODES = {
    'none': mine_none,
    'semi_hard_neg': mine_semi_hard_neg,
    'hard_neg': mine_hard_neg,
    'easy_neg': mine_easy_neg,
    'hard_pos': mine_hard_pos,
    'easy_pos': mine_easy_pos,
}
