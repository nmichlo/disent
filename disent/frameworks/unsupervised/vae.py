import torch
import torch.nn.functional as F


# ========================================================================= #
# Base VAE Loss                                                             #
# ========================================================================= #


class VaeLoss(object):

    def __call__(self, x, x_recon, z_mean, z_logvar, z_sampled, *args):
        return self.compute_loss(x, x_recon, z_mean, z_logvar, z_sampled, *args)

    @property
    def required_observations(self):
        """override in subclasses that need more observations, indicates format of arguments needed for compute_loss"""
        return 1

    def intercept_z(self, z_params, *args):
        """mutate z_mean and z_logvar before sampling"""
        return z_params, *args

    def compute_loss(self, forward_data, *args):
        """
        Compute the varous VAE loss components.
        Based on: https://github.com/google-research/disentanglement_lib/blob/a64b8b9994a28fafd47ccd866b0318fa30a3c76c/disentanglement_lib/methods/unsupervised/vae.py#L153
        """
        [(x, x_recon, (z_mean, z_logvar), z_sampled)] = (forward_data, *args)

        # reconstruction loss
        recon_loss = bce_loss_with_logits(x, x_recon)   # E[log p(x|z)]

        # regularizer
        kl_loss = kl_normal_loss(z_mean, z_logvar)      # D_kl(q(z|x) || p(z|x))
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)

        return {
            'train_loss': recon_loss + regularizer,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'elbo': -(recon_loss + kl_loss),
        }

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
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
