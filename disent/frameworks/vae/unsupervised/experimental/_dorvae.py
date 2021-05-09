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
from typing import Sequence

import torch
from torch.distributions import Normal
from typing import Optional

from disent.frameworks.helper.reconstructions import make_reconstruction_loss
from disent.frameworks.helper.reconstructions import ReconLossHandler
from disent.frameworks.helper.reductions import batch_loss_reduction
from disent.frameworks.vae.supervised import TripletVae
from disent.frameworks.vae.unsupervised._betavae import BetaVae
from disent.util.math_loss import torch_mse_rank_loss
from experiment.util.hydra_utils import instantiate_recursive


# ========================================================================= #
# tvae                                                                      #
# ========================================================================= #


class DataOverlapRankVae(TripletVae):

    REQUIRED_OBS = 1

    @dataclass
    class cfg(BetaVae.cfg):
        # tvae: no loss from decoder -> encoder
        detach: bool = False
        detach_decoder: bool = True
        detach_no_kl: bool = False
        detach_logvar: float = -2  # std = 0.5, logvar = ln(std**2) ~= -2,77
        # OVERLAP VAE
        overlap_loss: Optional[str] = None
        overlap_num: int = 1024
        # AUGMENT
        overlap_augment_mode: str = 'none'
        overlap_augment: Optional[dict] = None

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        # TODO: duplicate code
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
        # 2. generate random pairs -- this does not generate unique pairs
        a_idxs, p_idxs = torch.randint(len(x_targ), size=(2, self.cfg.overlap_num), device=x_targ.device)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # compute image distances
        with torch.no_grad():
            ap_recon_dists = self.overlap_handler.compute_pairwise_loss(x_targ[a_idxs], x_targ[p_idxs])
        # ++++++++++++++++++++++++++++++++++++++++++ #
        # compute representation distances
        ap_repr_dists = torch.abs(d_posterior.loc[a_idxs] - d_posterior.loc[p_idxs]).sum(dim=-1)
        # (z_sampled,) = zs_sampled
        # ap_repr_dists = torch.abs(z_sampled[a_idxs] - z_sampled[p_idxs]).sum(dim=-1)
        # ++++++++++++++++++++++++++++++++++++++++++ #
        loss = torch_mse_rank_loss(ap_repr_dists, ap_recon_dists.detach(), dims=-1, reduction='mean')
        return loss, {
            'mse_rank_loss': loss
        }

    def augment_triplet_targets(self, xs_targ):
        # TODO: duplicate code
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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
