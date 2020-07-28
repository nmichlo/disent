import torch

from disent.frameworks.unsupervised.betavae import BetaVaeLoss
from disent.frameworks.unsupervised.vae import bce_loss_with_logits, kl_normal_loss


# ========================================================================= #
# Averaging Functions                                                       #
# ========================================================================= #


def compute_average_gvae(z_mean, z_logvar, z2_mean, z2_logvar):
    """
    Compute the arithmetic mean of the encoder distributions.
    - Ada-GVAE Averaging function
    """
    # helper
    z_var, z2_var = z_logvar.exp(), z2_logvar.exp()

    # averages
    ave_var = (z_var + z2_var) * 0.5
    ave_mean = (z_mean + z2_mean) * 0.5

    # mean, logvar
    return ave_mean, ave_var.log()  # natural log

def compute_average_ml_vae(z_mean, z_logvar, z2_mean, z2_logvar):
    """
    Compute the product of the encoder distributions.
    - Ada-ML-VAE Averaging function
    """
    # helper
    z_var, z2_var = z_logvar.exp(), z2_logvar.exp()

    # Diagonal matrix inverse: E^-1 = 1 / E
    # https://proofwiki.org/wiki/Inverse_of_Diagonal_Matrix
    z_invvar, z2_invvar = z_var.reciprocal(), z2_var.reciprocal()

    # average var: E^-1 = E1^-1 + E2^-1
    ave_var = (z_invvar + z2_invvar).reciprocal()

    # average mean: u^T = (u1^T E1^-1 + u2^T E2^-1) E
    # u^T is horr vec (u is vert). E is square matrix
    ave_mean = (z_mean * z_invvar + z2_mean * z2_invvar) * ave_var

    # mean, logvar
    return ave_mean, ave_var.log()  # natural log


# ========================================================================= #
# Ada-GVAE                                                                  #
# ========================================================================= #


class AdaVaeLoss(BetaVaeLoss):

    AVERAGING_FUNCTIONS = {
        'ml-vae': compute_average_ml_vae,
        'gvae': compute_average_gvae,
    }

    def __init__(self, beta=4, average_mode='gvae'):
        super().__init__(beta=beta)
        # check averaging function
        if average_mode not in AdaVaeLoss.AVERAGING_FUNCTIONS:
            raise KeyError(f'Unsupported {average_mode=} must be one of: {set(AdaVaeLoss.AVERAGING_FUNCTIONS.keys())}')
        # set averaging function
        self.compute_average = AdaVaeLoss.AVERAGING_FUNCTIONS[average_mode]

    @property
    def required_observations(self):
        return 2

    def make_averaged(self, z_mean, z_logvar, z2_mean, z2_logvar, ave_mask):
        # compute average posteriors
        ave_mu, ave_logvar = self.compute_average(z_mean, z_logvar, z2_mean, z2_logvar)
        # modify estimated shared elements of original posteriors
        z_mean[ave_mask], z_logvar[ave_mask] = ave_mu[ave_mask], ave_logvar[ave_mask]
        z2_mean[ave_mask], z2_logvar[ave_mask] = ave_mu[ave_mask], ave_logvar[ave_mask]
        # return values
        return (z_mean, z_logvar), (z2_mean, z2_logvar)

    def intercept_z(self, z_params, *args):
        [(z_mean, z_logvar), (z2_mean, z2_logvar)] = (z_params, *args)

        # shared elements that need to be averaged, computed per pair in the batch.
        _, _, ave_mask = estimate_shared(z_mean, z_logvar, z2_mean, z2_logvar)

        # make averaged z parameters
        return self.make_averaged(z_mean, z_logvar, z2_mean, z2_logvar, ave_mask)

    def compute_loss(self, forward_data, *args):
        [(x, x_recon, (z_mean, z_logvar), z_sampled),
         (x2, x2_recon, (z2_mean, z2_logvar), z2_sampled)] = (forward_data, *args)

        # reconstruction error & KL divergence losses
        recon_loss = bce_loss_with_logits(x, x_recon)     # E[log p(x|z)]
        recon2_loss = bce_loss_with_logits(x2, x2_recon)  # E[log p(x|z)]
        kl_loss = kl_normal_loss(z_mean, z_logvar)        # D_kl(q(z|x) || p(z|x))
        kl2_loss = kl_normal_loss(z2_mean, z2_logvar)     # D_kl(q(z|x) || p(z|x))

        # compute combined loss
        # reduces down to summing the two BetaVAE losses
        loss = (recon_loss + recon2_loss) + self.beta * (kl_loss + kl2_loss)
        loss /= 2

        return {
            'train_loss': loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'elbo': -(recon_loss + kl_loss),
        }


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def kl_normal_loss_pair_elements(z_mean, z_logvar, z2_mean, z2_logvar):
    """Compute the KL divergence for normal distributions between all corresponding elements of a pair of latent vectors"""
    # compute GVAE deltas
    # σ0 = logv0.exp() ** 0.5
    # σ1 = logv1.exp() ** 0.5
    # return 0.5 * ((σ0/σ1)**2 + ((μ1 - μ0)**2)/(σ1**2) - 1 + 2*ln(σ1/σ0))
    # return 0.5 * ((σ0/σ1)**2 + ((μ1 - μ0)**2)/(σ1**2) - 1 + ln(σ1**2 / σ0**2))
    # return 0.5 * (σ0.exp()/σ1.exp() + (μ1 - μ0).pow(2)/σ1.exp() - 1 + (logv1 - logv0))
    return 0.5 * ((z_logvar.exp() / z2_logvar.exp()) + (z2_mean - z_mean).pow(2) / z2_logvar.exp() - 1 + (z2_logvar - z_logvar))

def estimate_kl_threshold(kl_deltas):
    """
    Compute the threshold for each image pair in a batch of kl divergences of all elements of the latent distributions.
    It should be noted that for a perfectly trained model, this threshold is always correct.
    """

    # Must return threshold for each image pair, not over entire batch.
    # specifying the axis returns a tuple of values and indices... better way?
    threshs = 0.5 * (kl_deltas.max(axis=1).values + kl_deltas.min(axis=1).values)
    return threshs[:, None]  # re-add the flattened dimension, shape=(batch_size, 1)

def estimate_shared(z_mean, z_logvar, z2_mean, z2_logvar):
    """
    Core of the adaptive VAE algorithm, estimating which factors
    have changed (or in this case which are shared and should remained unchanged
    by being be averaged) between pairs of observations.
    """
    # shared elements that need to be averaged, computed per pair in the batch.
    kl_deltas = kl_normal_loss_pair_elements(z_mean, z_logvar, z2_mean, z2_logvar)  # [𝛿_i ...]
    kl_threshs = estimate_kl_threshold(kl_deltas)  # threshold τ

    shared_mask = kl_deltas < kl_threshs  # true if 'unchanged' and should be average

    return kl_deltas, kl_threshs, shared_mask


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
