import torch
from disent.model import GaussianAutoEncoder
from disent.frameworks.framework import BaseFramework
from disent.frameworks.vae.loss import bce_loss_with_logits


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class AutoEncoder(BaseFramework):
    """
    Basic Auto Encoder
    """

    def __init__(self, make_optimizer_fn, make_model_fn):
        super().__init__(make_optimizer_fn)
        # vae model
        assert callable(make_model_fn)
        # TODO: convert to AE
        self._model: GaussianAutoEncoder = make_model_fn()
        assert isinstance(self._model, GaussianAutoEncoder)

    def compute_loss(self, batch, batch_idx):
        x = batch

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        z = self.encode(x)
        # reconstruct without the final activation
        x_recon = self.decode_partial(z)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon_loss = bce_loss_with_logits(x_recon, x)  # E[log p(x|z)]
        # compute combined loss
        loss = recon_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': recon_loss,
        }

    # --------------------------------------------------------------------- #
    # VAE Model Utility Functions (Visualisation)                           #
    # --------------------------------------------------------------------- #

    def encode(self, x):
        """Get the deterministic latent representation z = z_mean of observation x (useful for visualisation)"""
        z_mean, _ = self.encode_gaussian(x)
        return z_mean

    def decode(self, z):
        """Decode latent vector z into reconstruction x_recon (useful for visualisation)"""
        return self._model.reconstruct(z)

    def forward(self, batch) -> torch.Tensor:
        """The full deterministic model with the final activation (useful for visualisation)"""
        return self.decode(self.encode(batch))

    # --------------------------------------------------------------------- #
    # AE Model Utility Functions (Training)                                 #
    # --------------------------------------------------------------------- #

    def decode_partial(self, z):
        """Decode latent vector z into partial reconstruction x_recon, without the final activation (useful for training)"""
        return self._model.decode_partial(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
