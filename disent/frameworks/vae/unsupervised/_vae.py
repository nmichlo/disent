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
from typing import final
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
from torch.distributions import Distribution

from disent.frameworks.ae.unsupervised._ae import AE
from disent.frameworks.helper.latent_distributions import LatentDistsHandler
from disent.frameworks.helper.latent_distributions import make_latent_distribution
from disent.util import map_all


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class Vae(AE):
    """
    Variational Auto Encoder
    https://arxiv.org/abs/1312.6114
    """

    # override required z from AE
    REQUIRED_Z_MULTIPLIER = 2
    REQUIRED_OBS = 1

    @dataclass
    class cfg(AE.cfg):
        latent_distribution: str = 'normal'
        kl_loss_mode: str = 'direct'

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        # required_z_multiplier
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # vae distribution
        self.__latents_handler = make_latent_distribution(self.cfg.latent_distribution, kl_mode=self.cfg.kl_loss_mode, reduction=self.cfg.loss_reduction)

    @final
    @property
    def latents_handler(self) -> LatentDistsHandler:
        return self.__latents_handler

    # --------------------------------------------------------------------- #
    # VAE Training Step                                                     #
    # --------------------------------------------------------------------- #

    @final
    def do_training_step(self, batch, batch_idx):
        xs, xs_targ = self._get_xs_and_targs(batch, batch_idx)

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parameterizations
        zs_params = map_all(self.encode_params, xs)
        # [HOOK] intercept latent parameterizations
        zs_params, logs_intercept_zs = self.hook_intercept_zs(zs_params)
        # make latent distributions & sample
        ds_posterior, ds_prior, zs_sampled = map_all(self.params_to_dists_and_sample, zs_params, collect_returned=True)
        # [HOOK] intercept zs_samples
        zs_sampled, logs_intercept_zs_sampled = self.hook_intercept_zs_sampled(zs_sampled)
        # reconstruct without the final activation
        xs_partial_recon = map_all(self.decode_partial, zs_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # compute all the recon losses
        recon_loss, logs_recon = self.compute_ave_recon_loss(xs_partial_recon, xs_targ)
        # compute all the regularization losses
        reg_loss, logs_reg = self.compute_ave_reg_loss(ds_posterior, ds_prior, zs_sampled)
        # [HOOK] augment loss
        aug_loss, logs_aug = self.hook_compute_ave_aug_loss(ds_posterior=ds_posterior, ds_prior=ds_prior, zs_sampled=zs_sampled, xs_partial_recon=xs_partial_recon, xs_targ=xs_targ)
        # compute combined loss
        loss = recon_loss + reg_loss + aug_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # return values
        return loss, {
            **logs_intercept_zs,
            **logs_intercept_zs_sampled,
            **logs_recon,
            **logs_reg,
            **logs_aug,
            'recon_loss': recon_loss,
            'reg_loss': reg_loss,
            'aug_loss': aug_loss,
        }

    # --------------------------------------------------------------------- #
    # Overrideable                                                          #
    # --------------------------------------------------------------------- #

    def hook_intercept_zs(self, zs_params: Sequence['Params']) -> Tuple[Sequence['Params'], Dict[str, Any]]:
        return zs_params, {}

    def hook_intercept_zs_sampled(self, zs_sampled: Sequence[torch.Tensor]) -> Tuple[Sequence[torch.Tensor], Dict[str, Any]]:
        return zs_sampled, {}

    def hook_compute_ave_aug_loss(self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution], zs_sampled: Sequence[torch.Tensor], xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        return 0, {}

    def compute_ave_reg_loss(self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution], zs_sampled: Sequence[torch.Tensor]) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        # compute regularization loss (kl divergence)
        kl_loss = self.latents_handler.compute_ave_kl_loss(ds_posterior, ds_prior, zs_sampled)
        # return logs
        return kl_loss, {
            'kl_loss': kl_loss,
        }

    # --------------------------------------------------------------------- #
    # VAE - Encoding - Overrides AE                                         #
    # --------------------------------------------------------------------- #

    @final
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get the deterministic latent representation (useful for visualisation)"""
        z_params = self.encode_params(x)
        z = self.latents_handler.params_to_representation(z_params)
        return z

    @final
    def encode_params(self, x: torch.Tensor) -> 'Params':
        """Get parametrisations of the latent distributions, which are sampled from during training."""
        z_raw = self._model.encode(x)
        z_params = self.latents_handler.encoding_to_params(z_raw)
        return z_params

    # --------------------------------------------------------------------- #
    # VAE - Latent Distributions                                            #
    # --------------------------------------------------------------------- #

    @final
    def params_to_dists(self, z_params: 'Params') -> Tuple[Distribution, Distribution]:
        return self.latents_handler.params_to_dists(z_params)

    @final
    def params_to_dists_and_sample(self, z_params: 'Params') -> Tuple[Distribution, Distribution, torch.Tensor]:
        return self.latents_handler.params_to_dists_and_sample(z_params)

    @final
    def dist_to_params(self, d_posterior: Distribution) -> 'Params':
        return self.latents_handler.dist_to_params(d_posterior)

    @final
    def all_params_to_dists(self, zs_params: Sequence['Params']) -> Tuple[Sequence[Distribution], Sequence[Distribution]]:
        ds_posterior, ds_prior = zip(*(self.params_to_dists(z_params) for z_params in zs_params))
        return ds_posterior, ds_prior

    @final
    def all_dist_to_params(self, ds_posterior: Sequence[Distribution]) -> Sequence['Params']:
        return [self.dist_to_params(d) for d in ds_posterior]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

