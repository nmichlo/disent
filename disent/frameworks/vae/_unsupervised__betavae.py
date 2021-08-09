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
from numbers import Number
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
from torch.distributions import Distribution

from disent.frameworks.vae._unsupervised__vae import Vae


# ========================================================================= #
# Beta-VAE Loss                                                             #
# ========================================================================= #


class BetaVae(Vae):
    """
    beta-VAE: https://arxiv.org/abs/1312.6114
    """

    REQUIRED_OBS = 1

    @dataclass
    class cfg(Vae.cfg):
        # BETA SCALING:
        # =============
        # when using different loss reduction modes we need to scale beta to
        # preserve the ratio between loss components, by scaling beta.
        #   -- for loss_reduction='mean' we usually have:
        #      loss = mean_recon_loss + beta * mean_kl_loss
        #   -- for loss_reduction='mean_sum' we usually have:
        #      loss = (H*W*C) * mean_recon_loss + beta * (z_size) * mean_kl_loss
        #
        # So when switching from one mode to the other, we need to scale beta to
        # preserve these loss ratios:
        #   -- 'mean_sum' to 'mean':
        #      beta <- beta * (z_size) / (H*W*C)
        #   -- 'mean' to 'mean_sum':
        #      beta <- beta * (H*W*C) / (z_size)
        #
        # We obtain an equivalent beta for 'mean_sum' to 'mean':
        #   -- given values: beta=4 for 'mean_sum', with (H*W*C)=(64*64*3) and z_size=9
        #      beta = beta * ((z_size) / (H*W*C))
        #          ~= 4 * 0.0007324
        #          ~= 0,003
        #
        # This is similar to appendix A.6: `INTERPRETING NORMALISED β` of the beta-Vae paper:
        # - Published as a conference paper at ICLR 2017 (22 pages)
        # - https://openreview.net/forum?id=Sy2fzU9gl
        #
        beta: float = 0.003  # approximately equal to mean_sum beta of 4

    def __init__(self, model: 'AutoEncoder', cfg: cfg = None, batch_augment=None):
        super().__init__(model=model, cfg=cfg, batch_augment=batch_augment)
        assert self.cfg.beta >= 0, 'beta must be >= 0'

    # --------------------------------------------------------------------- #
    # Overrides                                                             #
    # --------------------------------------------------------------------- #

    def compute_ave_reg_loss(self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution], zs_sampled: Sequence[torch.Tensor]) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        # BetaVAE: compute regularization loss (kl divergence)
        kl_loss = self.latents_handler.compute_ave_kl_loss(ds_posterior, ds_prior, zs_sampled)
        kl_reg_loss = self.cfg.beta * kl_loss
        # return logs
        return kl_reg_loss, {
            'kl_loss': kl_loss,
            'kl_reg_loss': kl_reg_loss,
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
