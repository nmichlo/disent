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
from typing import final
from typing import Optional
from typing import Sequence

import torch
from torch.distributions import Normal

from disent.frameworks.helper.reconstructions import make_reconstruction_loss
from disent.frameworks.helper.reconstructions import ReconLossHandler
from disent.frameworks.vae._supervised__adaneg_tvae import AdaNegTripletVae
from disent.nn.loss.triplet_mining import configured_idx_mine


log = logging.getLogger(__name__)


# ========================================================================= #
# Mixin                                                                     #
# ========================================================================= #


class DataOverlapMixin(object):

    # should be inherited by the config on the child class
    @dataclass
    class cfg:
        # override from AE
        recon_loss: str = 'mse'
        # OVERLAP VAE
        overlap_loss: Optional[str] = None  # if None, use the value from recon_loss
        overlap_num: int = 1024
        overlap_mine_ratio: float = 0.1
        overlap_mine_triplet_mode: str = 'none'
        # AUGMENT
        overlap_augment_mode: str = 'augment'
        overlap_augment: Optional[dict] = None

    # private properties
    # - since this class does not have a constructor, it
    #   provides the `init_data_overlap_mixin` method, which
    #   should be called inside the constructor of the child class
    _augment: callable
    _overlap_handler: ReconLossHandler
    _init: bool

    def init_data_overlap_mixin(self):
        if hasattr(self, '_init'):
            raise RuntimeError(f'{DataOverlapMixin.__name__} on {self.__class__.__name__} was initialised more than once!')
        self._init = True
        # set augment and instantiate if needed
        if self.cfg.overlap_augment is not None:
            import hydra
            self._augment = hydra.utils.instantiate(self.cfg.overlap_augment)
            assert (self._augment is None) or callable(self._augment), f'augment is not None or callable: {repr(self._augment)}, obtained from `overlap_augment={repr(self.cfg.overlap_augment)}`'
        else:
            self._augment = None
        # get overlap loss
        if self.cfg.overlap_loss is None:
            log.info(f'`overlap_loss` not specified for {repr(self.__class__.__name__)}, using `recon_loss` instead: {repr(self.cfg.recon_loss)}')
            overlap_loss = self.cfg.recon_loss
        else:
            overlap_loss = self.cfg.overlap_loss
        # construct the overlap handler
        self._overlap_handler: ReconLossHandler = make_reconstruction_loss(name=overlap_loss, reduction='mean')

    @final
    @property
    def overlap_handler(self) -> ReconLossHandler:
        return self._overlap_handler

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

    @torch.no_grad()
    def augment_batch(self, x_targ):
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # perform the augments
        if self.cfg.overlap_augment_mode in ('augment', 'augment_each'):
            # recreate augment each time
            if self.cfg.overlap_augment_mode == 'augment_each':
                import hydra
                self._augment = hydra.utils.instantiate(self.cfg.overlap_augment)
            # augment on correct device, but skip if not defined!
            aug_x_targ = x_targ if (self._augment is None) else self._augment(x_targ)
        elif self.cfg.overlap_augment_mode == 'none':
            # no augment
            aug_x_targ = x_targ
        else:
            raise KeyError(f'invalid cfg.overlap_augment_mode={repr(self.cfg.overlap_augment_mode)}')
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # checks
        assert x_targ.shape == aug_x_targ.shape
        return aug_x_targ

    def mine_triplets(self, x_targ, a_idxs, p_idxs, n_idxs):
        return configured_idx_mine(
            x_targ=x_targ,
            a_idxs=a_idxs,
            p_idxs=p_idxs,
            n_idxs=n_idxs,
            cfg=self.cfg,
            pairwise_loss_fn=self.overlap_handler.compute_pairwise_loss,
        )

    def random_mined_triplets(self, x_targ_orig: torch.Tensor):
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # 1. augment batch
        aug_x_targ = self.augment_batch(x_targ_orig)
        # 2. generate random triples -- this does not generate unique pairs
        a_idxs, p_idxs, n_idxs = torch.randint(len(aug_x_targ), size=(3, min(self.cfg.overlap_num, len(aug_x_targ)**3)), device=aug_x_targ.device)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # self.debug(x_targ_orig, x_targ, a_idxs, p_idxs, n_idxs)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # TODO: this can be merged into a single function -- inefficient currently with deltas computed twice
        # 3. reorder random triples
        a_idxs, p_idxs, n_idxs = self.overlap_swap_triplet_idxs(aug_x_targ, a_idxs, p_idxs, n_idxs)
        # 4. mine random triples
        a_idxs, p_idxs, n_idxs = self.mine_triplets(aug_x_targ, a_idxs, p_idxs, n_idxs)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        return a_idxs, p_idxs, n_idxs

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


# ========================================================================= #
# Data Overlap Triplet VAE                                                  #
# ========================================================================= #


class DataOverlapTripletVae(AdaNegTripletVae, DataOverlapMixin):
    """
    Michlo et al.
    https://github.com/nmichlo/msc-research

    This is the unsupervised version of the Adaptive Triplet Loss for Variational Auto-Encoders
    - Triplets are sampled from training minibatches and are ordered using some distance
      function taken directly over the datapoints.
    - Adaptive Triplet Loss is used to guide distance learning and encourage disentanglement
    """

    REQUIRED_OBS = 1

    @dataclass
    class cfg(AdaNegTripletVae.cfg, DataOverlapMixin.cfg):
        pass

    def __init__(self, model: 'AutoEncoder', cfg: cfg = None, batch_augment=None):
        super().__init__(model=model, cfg=cfg, batch_augment=batch_augment)
        # initialise mixin
        self.init_data_overlap_mixin()

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Normal], ds_prior, zs_sampled, xs_partial_recon, xs_targ: Sequence[torch.Tensor]):
        [d_posterior], [x_targ_orig] = ds_posterior, xs_targ
        # 1. randomly generate and mine triplets using augmented versions of the inputs
        a_idxs, p_idxs, n_idxs = self.random_mined_triplets(x_targ_orig=x_targ_orig)
        # 2. compute triplet loss
        loss, loss_log = AdaNegTripletVae.estimate_ada_triplet_loss(
            ds_posterior=[Normal(d_posterior.loc[idxs], d_posterior.scale[idxs]) for idxs in (a_idxs, p_idxs, n_idxs)],
            cfg=self.cfg,
        )
        return loss, {
            **loss_log,
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
