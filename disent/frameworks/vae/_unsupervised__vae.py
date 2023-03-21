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
from typing import final

import torch
from torch.distributions import Distribution

from disent.frameworks._ae_mixin import _AeAndVaeMixin
from disent.frameworks.helper.latent_distributions import LatentDistsHandler
from disent.frameworks.helper.latent_distributions import make_latent_distribution
from disent.frameworks.helper.util import detach_all
from disent.util.iters import map_all

# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class Vae(_AeAndVaeMixin):
    """
    Variational Auto Encoder
    https://arxiv.org/abs/1312.6114
    ------------------------

    This VAE implementation supports multiple inputs in parallel. Each input
    is fed through the VAE on its own, and the loss from each step is averaged
    together at the end. This is effectively the same as increasing the batch
    size and scaling down the loss. The reason this VAE implementation supports
    the variable number of input args is to allow one common framework to be
    the parent of all child VAEs.

    Child classes can implement various hooks to override or add additional
    functionality to the VAE. Most common VAE derivatives can be implemented
    by simply adding functionality using these hooks, and changing the required
    number of input arguments using `REQUIRED_OBS`.

    - HOOKS:
        * `hook_intercept_ds`
        * `hook_compute_ave_aug_loss` (NB: not the same as `hook_ae_compute_ave_aug_loss` from AEs)

    - OVERRIDES:
        * `compute_ave_recon_loss`
        * `compute_ave_reg_loss`

    For example:
    -> implementing `hook_compute_ave_aug_loss` and setting `REQUIRED_OBS=3`
       we can easily implement a Triplet VAE.
    -> implementing `hook_intercept_ds` and then setting `REQUIRED_OBS=2`
       we can easily implement the Adaptive VAE style frameworks.

    TODO: allow hooks to be registered? simply build up new frameworks?
          Vae(recon_loss=DfcLoss())
          Vae(hook_intercept_ds=AdaptiveAveraging(mode='gvae'), required_obs=2)
          Vae(hook_compute_ave_aug_loss=TripletLoss(), required_obs=3)
    """

    # overrides
    REQUIRED_Z_MULTIPLIER = 2
    REQUIRED_OBS = 1

    @dataclass
    class cfg(_AeAndVaeMixin.cfg):
        # latent distribution settings
        latent_distribution: str = "normal"
        kl_loss_mode: str = "direct"
        # disable various components
        disable_reg_loss: bool = False

    def __init__(self, model: "AutoEncoder", cfg: cfg = None, batch_augment=None):
        # required_z_multiplier
        super().__init__(cfg=cfg, batch_augment=batch_augment)
        # initialise the auto-encoder mixin (recon handler, model, enc, dec, etc.)
        self._init_ae_mixin(model=model)
        # vae distribution
        self.__latents_handler = make_latent_distribution(
            self.cfg.latent_distribution, kl_mode=self.cfg.kl_loss_mode, reduction=self.cfg.loss_reduction
        )

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
        ds_posterior, ds_prior = map_all(self.encode_dists, xs, collect_returned=True)
        # [HOOK] intercept latent parameterizations
        ds_posterior, ds_prior, logs_intercept_ds = self.hook_intercept_ds(ds_posterior, ds_prior)
        # sample from dists
        zs_sampled = tuple(d.rsample() for d in ds_posterior)
        # reconstruct without the final activation
        xs_partial_recon = map_all(self.decode_partial, detach_all(zs_sampled, if_=self.cfg.detach_decoder))
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # compute all the recon losses
        recon_loss, logs_recon = self.compute_ave_recon_loss(xs_partial_recon, xs_targ)
        # compute all the regularization losses
        reg_loss, logs_reg = self.compute_ave_reg_loss(ds_posterior, ds_prior, zs_sampled)
        # [HOOK] augment loss
        aug_loss, logs_aug = self.hook_compute_ave_aug_loss(
            ds_posterior=ds_posterior,
            ds_prior=ds_prior,
            zs_sampled=zs_sampled,
            xs_partial_recon=xs_partial_recon,
            xs_targ=xs_targ,
        )
        # compute combined loss
        loss = 0
        if not self.cfg.disable_rec_loss:
            loss += recon_loss
        if not self.cfg.disable_aug_loss:
            loss += aug_loss
        if not self.cfg.disable_reg_loss:
            loss += reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # log general
        self.log_dict(
            {
                **logs_intercept_ds,
                **logs_recon,
                **logs_reg,
                **logs_aug,
            }
        )

        # log progress bar
        self.log_dict(
            {
                "recon_loss": float(recon_loss),
                "reg_loss": float(reg_loss),
                "aug_loss": float(aug_loss),
            },
            prog_bar=True,
        )

        # return values
        return loss

    # --------------------------------------------------------------------- #
    # Overrideable Hooks                                                    #
    # --------------------------------------------------------------------- #

    def hook_intercept_ds(
        self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution]
    ) -> Tuple[Sequence[Distribution], Sequence[Distribution], Dict[str, Any]]:
        return ds_posterior, ds_prior, {}

    def hook_compute_ave_aug_loss(
        self,
        ds_posterior: Sequence[Distribution],
        ds_prior: Sequence[Distribution],
        zs_sampled: Sequence[torch.Tensor],
        xs_partial_recon: Sequence[torch.Tensor],
        xs_targ: Sequence[torch.Tensor],
    ) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        return 0, {}

    def compute_ave_recon_loss(
        self, xs_partial_recon: Sequence[torch.Tensor], xs_targ: Sequence[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        # compute reconstruction loss
        pixel_loss = self.recon_handler.compute_ave_loss_from_partial(xs_partial_recon, xs_targ)
        # return logs
        return pixel_loss, {"pixel_loss": pixel_loss}

    def compute_ave_reg_loss(
        self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution], zs_sampled: Sequence[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, Number], Dict[str, Any]]:
        # compute regularization loss (kl divergence)
        kl_loss = self.latents_handler.compute_ave_kl_loss(ds_posterior, ds_prior, zs_sampled)
        # return logs
        return kl_loss, {
            "kl_loss": kl_loss,
        }

    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Visualisation)                           #
    # --------------------------------------------------------------------- #

    @final
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get the deterministic latent representation (useful for visualisation)"""
        z_raw = self._model.encode(x, chunk=True)
        z = self.latents_handler.encoding_to_representation(z_raw)
        return z

    @final
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into reconstruction x_recon (useful for visualisation)"""
        return self.recon_handler.activate(self._model.decode(z))

    @final
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Feed through the full deterministic model (useful for visualisation)"""
        return self.decode(self.encode(batch))

    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Training)                                #
    # --------------------------------------------------------------------- #

    @final
    def encode_dists(self, x: torch.Tensor) -> Tuple[Distribution, Distribution]:
        """Get parametrisations of the latent distributions, which are sampled from during training."""
        z_raw = self._model.encode(x, chunk=True)
        z_posterior, z_prior = self.latents_handler.encoding_to_dists(z_raw)
        return z_posterior, z_prior

    @final
    def decode_partial(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector z into partial reconstructions that exclude the final activation if there is one."""
        return self._model.decode(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
