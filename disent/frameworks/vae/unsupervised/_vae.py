from dataclasses import dataclass
from typing import Tuple

import torch
from torch.distributions import Distribution

from disent.frameworks.ae.unsupervised import AE
from disent.distributions.vae import make_vae_distribution, VaeDistribution

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
        distribution: str = 'normal'
        kl_mode: str = 'direct'

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: cfg = None):
        # required_z_multiplier
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, cfg=cfg)
        # vae distribution
        self._distributions: VaeDistribution = make_vae_distribution(self.cfg.distribution)

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
        (d_posterior, d_prior), z_sampled = self.training_make_distributions_and_sample(z_params)
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get the deterministic latent representation (useful for visualisation)"""
        z = self._distributions.params_to_z(self.training_encode_params(x))
        return z

    def training_encode_params(self, x: torch.Tensor) -> 'Params':
        """Get parametrisations of the latent distributions, which are sampled from during training."""
        z_params = self._distributions.raw_to_params(self._model.encode(x))
        return z_params

    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Training)                                #
    # --------------------------------------------------------------------- #

    def training_make_distributions_and_sample(self, z_params: 'Params') -> Tuple[Tuple[Distribution, Distribution], torch.Tensor]:
        return self._distributions.make_and_sample(z_params)

    def training_make_distributions(self, z_params: 'Params') -> Tuple[Distribution, Distribution]:
        return self._distributions.make(z_params)

    def training_kl_loss(self, d_posterior: Distribution, d_prior: Distribution, z_sampled: torch.Tensor = None) -> torch.Tensor:
        if self.cfg.kl_mode == 'direct':
            # This is how the original VAE/BetaVAE papers do it:s
            # - we compute the kl divergence directly instead of approximating it
            kl = torch.distributions.kl_divergence(d_posterior, d_prior)
        elif self.cfg.kl_mode == 'approx':
            # This is how pytorch-lightning-bolts does it:
            # See issue: https://github.com/PyTorchLightning/pytorch-lightning-bolts/issues/565
            # - we approximate the kl divergence instead of computing it analytically
            assert z_sampled is not None, 'to compute the approximate kl loss, z_sampled needs to be defined (cfg.kl_mode="approx")'
            kl = d_posterior.log_prob(z_sampled) - d_prior.log_prob(z_sampled)
        else:
            raise KeyError(f'invalid kl_mode={repr(self.cfg.kl_mode)}')
        # average and scale
        kl = kl.mean() * self._model.z_size
        return kl

    def training_regularize_kl(self, kl_loss):
        return kl_loss


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
