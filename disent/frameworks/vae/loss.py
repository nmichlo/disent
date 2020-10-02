import torch
import torch.nn.functional as F


# ========================================================================= #
# Vae Loss Functions                                                        #
# ========================================================================= #

def bce_loss_with_logits(x_recon, x_target):
    """
    Computes the Bernoulli loss for the sigmoid activation function
    FROM: https://github.com/google-research/disentanglement_lib/blob/76f41e39cdeff8517f7fba9d57b09f35703efca9/disentanglement_lib/methods/shared/losses.py
    """
    # x, x_recon = x.view(x.shape[0], -1), x_recon.view(x.shape[0], -1)
    # per_sample_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none').sum(axis=1)  # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x), axis=1)
    # reconstruction_loss = per_sample_loss.mean()                                                    # tf.reduce_mean(per_sample_loss)
    # ALTERNATIVE IMPLEMENTATION https://github.com/YannDubs/disentangling-vae/blob/master/disvae/models/losses.py
    assert x_recon.shape == x_target.shape
    return F.binary_cross_entropy_with_logits(x_recon, x_target, reduction="sum") / len(x_target)

def kl_normal_loss(mu, logvar):
    """
    Calculates the KL divergence between a normal distribution with
    diagonal covariance and a unit normal distribution.
    FROM: https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py

    (✓) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (compute_gaussian_kl)
    """
    # Calculate KL divergence
    kl_values = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Sum KL divergence across latent vector for each sample
    kl_sums = torch.sum(kl_values, dim=1)
    # KL loss is mean of the KL divergence sums
    kl_loss = torch.mean(kl_sums)
    return kl_loss

# ========================================================================= #
# END                                                                       #
# ========================================================================= #

