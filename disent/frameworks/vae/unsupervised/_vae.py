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
from typing import Dict
from typing import final
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
from torch.distributions import Distribution

from disent.frameworks.ae.unsupervised import AE
from disent.frameworks.helper.latent_distributions import LatentDistsHandler
from disent.frameworks.helper.latent_distributions import make_latent_distribution


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #
from disent.util import aggregate_dict
from disent.util import collect_dicts
from disent.util import map_all


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
        self._latents_handler: LatentDistsHandler = make_latent_distribution(self.cfg.latent_distribution)

    # --------------------------------------------------------------------- #
    # VAE Training Step                                                     #
    # --------------------------------------------------------------------- #

    @final
    def do_training_step(self, batch, batch_idx):
        xs, xs_targ = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parameterizations
        zs_params = map_all(self.encode_params, xs)
        # intercept latent parameterizations
        zs_params, logs_intercept = self.training_intercept_zs(zs_params)
        # make latent distributions & sample
        ds_posterior, ds_prior, zs_sampled = map_all(self.params_to_dists_and_sample, zs_params, collect_returned=True)
        # reconstruct without the final activation
        xs_partial_recon = map_all(self.decode_partial, zs_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # compute all the recon losses
        recon_loss, logs_recon = map_all(self.compute_reconstruction_loss, xs_partial_recon, xs_targ, collect_returned=True)
        recon_loss = sum(recon_loss) / len(recon_loss)
        # compute all the regularization losses
        reg_loss, logs_reg = map_all(self.compute_regularization_loss, ds_posterior, ds_prior, zs_sampled, collect_returned=True)
        reg_loss = sum(reg_loss) / len(reg_loss)
        # compute combined loss
        loss = recon_loss + reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # return values
        return loss, {
            **logs_intercept,
            **aggregate_dict(collect_dicts(logs_recon)),
            **aggregate_dict(collect_dicts(logs_reg)),
            'recon_loss': recon_loss,
            'reg_loss': reg_loss,
        }

    # --------------------------------------------------------------------- #
    # Overrideable                                                          #
    # --------------------------------------------------------------------- #

    def training_intercept_zs(self, zs_params: Sequence['Params']) -> Sequence['Params']:
        return zs_params

    def compute_regularization_loss(self, d_posterior: Distribution, d_prior: Distribution, z_sampled: torch.Tensor) -> (torch.Tensor, Dict[str, float]):
        # compute regularization loss (kl divergence)
        kl_loss = self.compute_kl_loss(d_posterior, d_prior, z_sampled)
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
        z = self._latents_handler.params_to_representation(z_params)
        return z

    @final
    def encode_params(self, x: torch.Tensor) -> 'Params':
        """Get parametrisations of the latent distributions, which are sampled from during training."""
        z_raw = self._model.encode(x)
        z_params = self._latents_handler.encoding_to_params(z_raw)
        return z_params

    # --------------------------------------------------------------------- #
    # VAE - Latent Distributions                                            #
    # --------------------------------------------------------------------- #

    @final
    def params_to_dists_and_sample(self, z_params: 'Params') -> Tuple[Distribution, Distribution, torch.Tensor]:
        return self._latents_handler.params_to_dists_and_sample(z_params)

    @final
    def compute_kl_loss(self, d_posterior: Distribution, d_prior: Distribution, z_sampled: Optional[torch.Tensor]) -> torch.Tensor:
        return self._latents_handler.compute_kl_loss(
            d_posterior, d_prior, z_sampled,
            mode=self.cfg.kl_loss_mode,
            reduction=self.cfg.loss_reduction,
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #



# class MultiVae(Vae):
#     """
#     Variational Auto Encoder
#     https://arxiv.org/abs/1312.6114
#     """
#
#     # override required z from AE
#     REQUIRED_Z_MULTIPLIER = 2
#
#     @dataclass
#     class cfg(AE.cfg):
#         latent_distribution: str = 'normal'
#         kl_loss_mode: str = 'direct'
#
#     def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
#         # required_z_multiplier
#         super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
#         # vae distribution
#         self._distributions: LatentDistribution = make_latent_distribution(self.cfg.latent_distribution)
#
#     # --------------------------------------------------------------------- #
#     # VAE Training Step -- Overridable                                      #
#     # --------------------------------------------------------------------- #
#
#     def compute_training_loss(self, batch, batch_idx):
#         xs, xs_targ = batch['x'], batch['x_targ']
#
#         # FORWARD
#         # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
#         # latent distribution parameterizations
#         zs_params = self.training_encode_params(xs)
#         # make latent distributions & sample
#         ds_posterior, ds_prior = self.training_params_to_dists(zs_params)
#         zs_sampled = self.training_sample_representation(ds_posterior)
#         # reconstruct without the final activation
#         xs_partial_recon = self.training_decode_partial(zs_sampled)
#         # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
#
#         # LOSS
#         # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
#         # reconstruction loss
#         ave_recon_loss, logs_recon = self.compute_reconstruction_loss(xs_partial_recon, xs_targ)  # E[log p(x|z)]
#         # regularization loss (kl divergence)
#         ave_reg_loss, logs_reg = self.compute_regularization_loss(ds_posterior, ds_prior, zs_sampled)  # D_kl(q(z|x) || p(z|x))
#         # compute combined loss
#         ave_loss = ave_recon_loss + ave_reg_loss
#         # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
#
#         return {
#             'train_loss': loss,
#             'recon_loss': recon_loss,
#             'reg_loss': reg_loss,
#             # extra logs
#             **logs_recon,
#             **logs_reg,
#             'elbo': -(logs_recon['pixel_loss'] + logs_reg['kl_loss']),
#         }
#
#     # --------------------------------------------------------------------- #
#     # VAE - Multiple                                                        #
#     # --------------------------------------------------------------------- #




