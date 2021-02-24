from dataclasses import dataclass
from typing import Tuple, final

import torch
from torch.distributions import Distribution

from disent.frameworks.ae.unsupervised import AE
from disent.frameworks.helper.latent_distributions import make_latent_distribution, LatentDistribution


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

    @dataclass
    class cfg(AE.cfg):
        latent_distribution: str = 'normal'
        kl_loss_mode: str = 'direct'

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        # required_z_multiplier
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # vae distribution
        self._distributions: LatentDistribution = make_latent_distribution(self.cfg.latent_distribution)

    # --------------------------------------------------------------------- #
    # VAE Training Step                                                     #
    # --------------------------------------------------------------------- #

    def compute_training_loss(self, batch, batch_idx):
        (x,), (x_targ,) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parameterizations
        z_params = self.training_encode_params(x)
        # sample from latent distribution
        (d_posterior, d_prior), z_sampled = self.training_params_to_distributions_and_sample(z_params)
        # reconstruct without the final activation
        x_partial_recon = self.training_decode_partial(z_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon_loss = self.training_recon_loss(x_partial_recon, x_targ)  # E[log p(x|z)]
        # KL divergence & regularization
        kl_loss = self.training_kl_loss(d_posterior, d_prior, z_sampled)  # D_kl(q(z|x) || p(z|x))
        # compute kl regularisation
        kl_reg_loss = self.training_regularize_kl(kl_loss)
        # compute combined loss
        loss = recon_loss + kl_reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': recon_loss,
            'kl_reg_loss': kl_reg_loss,
            'kl_loss': kl_loss,
            'elbo': -(recon_loss + kl_loss),
        }

    # --------------------------------------------------------------------- #
    # VAE - Overrides AE                                                    #
    # --------------------------------------------------------------------- #

    @final
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get the deterministic latent representation (useful for visualisation)"""
        z = self._distributions.params_to_representation(self.training_encode_params(x))
        return z

    @final
    def training_encode_params(self, x: torch.Tensor) -> 'Params':
        """Get parametrisations of the latent distributions, which are sampled from during training."""
        z_params = self._distributions.encoding_to_params(self._model.encode(x))
        return z_params

    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Training)                                #
    # --------------------------------------------------------------------- #

    @final
    def training_params_to_distributions_and_sample(self, z_params: 'Params') -> Tuple[Tuple[Distribution, Distribution], torch.Tensor]:
        return self._distributions.params_to_distributions_and_sample(z_params)

    @final
    def training_params_to_distributions(self, z_params: 'Params') -> Tuple[Distribution, Distribution]:
        return self._distributions.params_to_distributions(z_params)

    @final
    def training_kl_loss(self, d_posterior: Distribution, d_prior: Distribution, z_sampled: torch.Tensor = None) -> torch.Tensor:
        return self._distributions.compute_kl_loss(
            d_posterior, d_prior, z_sampled,
            mode=self.cfg.kl_loss_mode,
            reduction=self.cfg.loss_reduction,
        )

    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Overridable)                             #
    # --------------------------------------------------------------------- #

    def training_regularize_kl(self, kl_loss):
        return kl_loss


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

