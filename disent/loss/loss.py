import torch
import torch.nn.functional as F
from disent.math import anneal_step


# ========================================================================= #
# Vae Loss                                                                  #
# ========================================================================= #


def _bce_loss_with_logits(x, x_recon):
    """
    Computes the Bernoulli loss for the sigmoid activation function
    FROM: https://github.com/google-research/disentanglement_lib/blob/76f41e39cdeff8517f7fba9d57b09f35703efca9/disentanglement_lib/methods/shared/losses.py
    """
    x, x_recon = x.view(x.shape[0], -1), x_recon.view(x.shape[0], -1)
    per_sample_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none').sum(axis=1)  # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x), axis=1)
    reconstruction_loss = per_sample_loss.mean()                                                    # tf.reduce_mean(per_sample_loss)
    return reconstruction_loss  # F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum') / x.shape[0]

def _bce_loss(x, x_recon):
    """Computes the Bernoulli loss"""
    return F.binary_cross_entropy(x_recon, x, reduction='none').sum(axis=1).mean()


def _kl_normal_loss(mu, logvar):
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

class VaeLoss(object):

    def __call__(self, x, x_recon, z_mean, z_logvar, z_sampled):
        return self.compute_loss(x, x_recon, z_mean, z_logvar, z_sampled)

    def compute_loss(self, x, x_recon, z_mean, z_logvar, z_sampled):
        """
        Compute the varous VAE loss components.
        Based on: https://github.com/google-research/disentanglement_lib/blob/a64b8b9994a28fafd47ccd866b0318fa30a3c76c/disentanglement_lib/methods/unsupervised/vae.py#L153
        """
        # reconstruction loss
        recon_loss = _bce_loss_with_logits(x, x_recon)   # E[log p(x|z)]

        # regularizer
        kl_loss = _kl_normal_loss(z_mean, z_logvar)      # D_kl(q(z|x) || p(z|x))
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)

        return {
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'loss': recon_loss + regularizer,
            'elbo': -(recon_loss + kl_loss),
        }

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        return kl_loss


# ========================================================================= #
# Beta-VAE Loss                                                             #
# ========================================================================= #

class BetaVaeLoss(VaeLoss):
    def __init__(self, beta=4):
        self.beta = beta

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        return self.beta * kl_loss


class BetaVaeHLoss(BetaVaeLoss):
    """
    Compute the Beta-VAE loss as in [1]

    [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
    a constrained variational framework." (2016).
    """

    def __init__(self, beta=4, anneal_end_steps=None):
        super().__init__(beta)
        self.n_train_steps = 0
        self.anneal_end_steps = anneal_end_steps  # TODO

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        print('WARNING: training steps not updated')
        anneal_reg = anneal_step(0, 1, self.n_train_steps, self.anneal_end_steps) # if is_train else 1
        return (anneal_reg * self.beta) * kl_loss


# ========================================================================= #
# Ada-GVae Loss                                                             #
# ========================================================================= #

def _kl_normal_loss_pair_elements(z_mean, z_logvar, z2_mean, z2_logvar):
    """Compute the KL divergence for normal distributions between all corresponding elements of a pair of latent vectors"""
    # compute GVAE deltas
    # σ0 = logv0.exp() ** 0.5
    # σ1 = logv1.exp() ** 0.5
    # return 0.5 * ((σ0/σ1)**2 + ((μ1 - μ0)**2)/(σ1**2) - 1 + 2*ln(σ1/σ0))
    # return 0.5 * (σ0.exp()/σ1.exp() + (μ1 - μ0).pow(2)/σ1.exp() - 1 + (logv1 - logv0))
    return 0.5 * (z_logvar.exp() / z2_logvar.exp() + (z2_mean - z_mean).pow(2) / z2_logvar.exp() - 1 + (z_logvar - z_logvar))

def _estimate_kl_threshold(kl_deltas):
    """
    Compute the threshold for corresponding elements of the latent vectors that are unchanged between a sample pair.
    It should be noted that for a perfectly trained model, this threshold is always correct.
    """
    return 0.5 * (kl_deltas.max() + kl_deltas.min())

class AdaGVaeLoss(BetaVaeLoss):

    def __init__(self, vae, sampler, beta=4):
        super().__init__(beta)
        self.sampler = sampler
        self.vae = vae

    def compute_loss(self, x, x_recon, z_mean, z_logvar, is_train):
        # generate new pair
        x2 = self.sampler()
        x2_recon, z2_mean, z2_logvar = self.vae(x2)
        # TODO: this is a batch, not a single item
        # TODO: calculate threshold per pair not over entire batch

        # shared elements that need to be averaged
        kl_deltas = _kl_normal_loss_pair_elements(z_mean, z_logvar, z2_mean, z2_logvar)  # [𝛿_i ...]
        kl_thresh = _estimate_kl_threshold(kl_deltas)                                # threshold τ
        ave_elements = kl_deltas < kl_thresh
        # TODO: do you average distributions or do you average samples from distributions? I think the former.
        # compute average posteriors
        # TODO: is this correct?
        # TODO: is this AdaGVAE or AdaMLVae?
        ave_mu, ave_logvar = (z_mean + z2_mean) * 0.5, (z_logvar + z2_logvar) * 0.5
        # compute approximate posteriors
        # approx_z_mean, approx_z_logvar = z_mean.clone(), z_logvar.clone()
        # approx_z2_mean, approx_z2_logvar = z2_mean.clone(), z2_logvar.clone()
        z_mean[ave_elements], z_logvar[ave_elements] = ave_mu[ave_elements], ave_logvar[ave_elements]
        z2_mean[ave_elements], z2_logvar[ave_elements] = ave_mu[ave_elements], ave_logvar[ave_elements]

        # TODO: x_recon and x2_recon need to use updated/averaged z
        # reconstruction error & KL divergence losses
        recon_loss = _bce_loss(x, x_recon)            # E[log p(x|z)]
        recon2_loss = _bce_loss(x2, x2_recon)         # E[log p(x|z)]
        kl_loss = _kl_normal_loss(z_mean, z_logvar)     # D_kl(q(z|x) || p(z|x))
        kl2_loss = _kl_normal_loss(z2_mean, z2_logvar)  # D_kl(q(z|x) || p(z|x))

        # compute combined loss
        return (recon_loss + recon2_loss) + self.beta * (kl_loss + kl2_loss)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
