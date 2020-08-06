import torch
import torch.nn.functional as F
from disent.frameworks.framework import BaseFramework
from disent.model import GaussianAutoEncoder


# ========================================================================= #
# framework_vae                                                             #
# ========================================================================= #


class Vae(BaseFramework):
    """
    Variational Auto Encoder
    https://arxiv.org/abs/1312.6114
    """

    def __init__(self, make_optimizer_fn, make_model_fn):
        super().__init__(make_optimizer_fn)
        # vae model
        assert callable(make_model_fn)
        self.model = make_model_fn()
        assert isinstance(self.model, GaussianAutoEncoder)

    def forward(self, batch) -> torch.Tensor:
        return self.model.forward_deterministic(batch)

    def compute_loss(self, batch, batch_idx):
        x = batch

        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        z_mean, z_logvar = self.model.encode_gaussian(x)
        # sample from latent distribution
        z = self.model.reparameterize(z_mean, z_logvar)
        # reconstruct without the final activation
        x_recon = self.model.decode(z)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon_loss = bce_loss_with_logits(x, x_recon)  # E[log p(x|z)]
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


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def bce_loss_with_logits(x, x_recon):
    """
    Computes the Bernoulli loss for the sigmoid activation function
    FROM: https://github.com/google-research/disentanglement_lib/blob/76f41e39cdeff8517f7fba9d57b09f35703efca9/disentanglement_lib/methods/shared/losses.py
    """
    # x, x_recon = x.view(x.shape[0], -1), x_recon.view(x.shape[0], -1)
    # per_sample_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none').sum(axis=1)  # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x), axis=1)
    # reconstruction_loss = per_sample_loss.mean()                                                    # tf.reduce_mean(per_sample_loss)
    # ALTERNATIVE IMPLEMENTATION https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py
    batch_size = x.shape[0]
    reconstruction_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction="sum") / batch_size
    # return
    return reconstruction_loss

def kl_normal_loss(mu, logvar):
    """
    Calculates the KL divergence between a normal distribution with
    diagonal covariance and a unit normal distribution.
    FROM: https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py
    """
    # Calculate KL divergence
    kl_values = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Mean KL divergence across batch for each latent variable
    kl_means = torch.mean(kl_values, dim=0)
    # KL loss is sum of mean KL of each latent variable
    kl_loss = torch.sum(kl_means)
    return kl_loss


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
