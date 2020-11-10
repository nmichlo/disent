import torch
from dataclasses import dataclass
from disent.frameworks.framework import BaseFramework
from disent.frameworks.vae.loss import bce_loss_with_logits, kl_normal_loss
from disent.model.ae.base import GaussianAutoEncoder


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class Vae(BaseFramework):
    """
    Variational Auto Encoder
    https://arxiv.org/abs/1312.6114
    """

    def __init__(self, make_optimizer_fn, make_model_fn, batch_augment=None, cfg: Config = Config()):
        super().__init__(make_optimizer_fn, batch_augment=batch_augment, cfg=cfg)
        # vae model
        assert callable(make_model_fn)
        self._model: GaussianAutoEncoder = make_model_fn()
        assert isinstance(self._model, GaussianAutoEncoder)

    def compute_training_loss(self, batch, batch_idx):
        (x,), (x_targ,) = batch['x'], batch['x_targ']

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        z_mean, z_logvar = self.encode_gaussian(x)
        # sample from latent distribution
        z = self.reparameterize(z_mean, z_logvar)
        # reconstruct without the final activation
        x_recon = self.decode_partial(z)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon_loss = bce_loss_with_logits(x_recon, x_targ)  # E[log p(x|z)]
        # KL divergence
        kl_loss = kl_normal_loss(z_mean, z_logvar)     # D_kl(q(z|x) || p(z|x))
        # compute kl regularisation
        kl_reg_loss = self.kl_regularization(kl_loss)
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

    def kl_regularization(self, kl_loss):
        return kl_loss

    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Visualisation)                           #
    # --------------------------------------------------------------------- #

    def encode(self, x):
        """Get the deterministic latent representation z = z_mean of observation x (useful for visualisation)"""
        z_mean, _ = self.encode_gaussian(x)
        return z_mean

    def decode(self, z):
        """Decode latent vector z into reconstruction x_recon (useful for visualisation)"""
        return self._model.decode(z)

    def forward(self, batch) -> torch.Tensor:
        """The full deterministic model with the final activation (useful for visualisation)"""
        return self.decode(self.encode(batch))

    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Training)                                #
    # --------------------------------------------------------------------- #

    def encode_gaussian(self, x):
        """Get latent parametrisation (z_mean, z_logvar) of observation x (useful for training)"""
        return self._model.encode_gaussian(x)

    def reparameterize(self, z_mean, z_logvar):
        """
        Randomly sample for z based on the parametrization of the gaussian normal with diagonal covariance.
        This is an implementation of the 'reparameterization trick'.
        ie. z ~ p(z|x)
        Gaussian Encoder Model Distribution - pg. 25 in Variational Auto Encoders

        (✓) Visual inspection against reference implementation:
            https://github.com/google-research/disentanglement_lib (sample_from_latent_distribution)
        """
        std = torch.exp(0.5 * z_logvar)  # std == var^0.5 == e^(log(var^0.5)) == e^(0.5*log(var))
        eps = torch.randn_like(std)      # N(0, 1)
        return z_mean + (std * eps)      # mu + dot(std, eps)

    def decode_partial(self, z):
        """Decode latent vector z into partial reconstruction x_recon, without the final activation (useful for training)"""
        return self._model.decode_partial(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
