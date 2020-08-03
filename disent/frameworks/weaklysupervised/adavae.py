import gin
import torch

from disent.frameworks.framework import BaseFramework
from disent.model import GaussianAutoEncoder
from disent.frameworks.unsupervised.vae import TrainingData, bce_loss_with_logits, kl_normal_loss


# ========================================================================= #
# Ada-GVAE                                                                  #
# ========================================================================= #


@gin.configurable('framework.weaklysupervised.AdaVae')
class AdaVae(BaseFramework):
    
    AVE_MODE_GVAE = 'gvae'
    AVE_MODE_ML_VAE = 'ml-vae'
    
    AVE_MODES = {AVE_MODE_GVAE, AVE_MODE_ML_VAE}
    
    def __init__(self, beta=4, average_mode=AVE_MODE_GVAE):
        # set averaging function
        if average_mode == AdaVae.AVE_MODE_GVAE:
            self.compute_average = compute_average_gvae
        elif average_mode == AdaVae.AVE_MODE_ML_VAE:
            self.compute_average = compute_average_ml_vae
        else:
            raise KeyError(f'Invalid averaging mode: {repr(average_mode)}, must be one of: {AdaVae.AVE_MODES}')
        # beta-vae params
        self.beta = beta

    def training_step(self, model: GaussianAutoEncoder, batch):
        x0, x1 = batch
        # ENCODE
        z0_mean, z0_logvar = model.encode_gaussian(x0)
        z1_mean, z1_logvar = model.encode_gaussian(x1)
        # INTERCEPT
        (z0_mean, z0_logvar, z1_mean, z1_logvar), intercept_logs = self.intercept_z(z0_mean, z0_logvar, z1_mean, z1_logvar)
        # REPARAMETERIZE
        z0_sampled = model.reparameterize(z0_mean, z0_logvar)
        z1_sampled = model.reparameterize(z1_mean, z1_logvar)
        # RECONSTRUCT
        x0_recon = model.decode(z0_sampled)
        x1_recon = model.decode(z1_sampled)
        # COMPUTE LOSS
        loss_logs = self.compute_loss(
            TrainingData(x0, x0_recon, z0_mean, z0_logvar, z0_sampled),
            TrainingData(x1, x1_recon, z1_mean, z1_logvar, z1_sampled),
        )
        # RETURN INFO
        return {
            **intercept_logs,
            **loss_logs,
        }
    
    def intercept_z(self, z0_mean, z0_logvar, z1_mean, z1_logvar):
        # shared elements that need to be averaged, computed per pair in the batch.
        _, _, share_mask = estimate_shared(z0_mean, z0_logvar, z1_mean, z1_logvar)
        # make averaged z parameters
        new_args = self.make_averaged(z0_mean, z0_logvar, z1_mean, z1_logvar, share_mask)
        # return new args & generate logs
        return new_args, {'shared': share_mask.sum(dim=1).mean()}

    def make_averaged(self, z0_mean, z0_logvar, z1_mean, z1_logvar, share_mask):
        # compute average posteriors
        ave_mu, ave_logvar = self.compute_average(z0_mean, z0_logvar, z1_mean, z1_logvar)
        # modify estimated shared elements of original posteriors
        z0_mean[share_mask], z0_logvar[share_mask] = ave_mu[share_mask], ave_logvar[share_mask]
        z1_mean[share_mask], z1_logvar[share_mask] = ave_mu[share_mask], ave_logvar[share_mask]
        # return values
        return z0_mean, z0_logvar, z1_mean, z1_logvar

    def compute_loss(self, data0: TrainingData, data1: TrainingData):
        x0, x0_recon, z0_mean, z0_logvar, z0_sampled = data0
        x1, x1_recon, z1_mean, z1_logvar, z1_sampled = data1
        
        # reconstruction error
        recon0_loss = bce_loss_with_logits(x0, x0_recon)  # E[log p(x|z)]
        recon1_loss = bce_loss_with_logits(x1, x1_recon)  # E[log p(x|z)]
        ave_recon_loss = (recon0_loss + recon1_loss) / 2

        # KL divergence
        kl0_loss = kl_normal_loss(z0_mean, z0_logvar)     # D_kl(q(z|x) || p(z|x))
        kl1_loss = kl_normal_loss(z1_mean, z1_logvar)     # D_kl(q(z|x) || p(z|x))
        ave_kl_loss = (kl0_loss + kl1_loss) / 2

        # compute combined loss - must be same as the BetaVAE
        loss = ave_recon_loss + (self.beta * ave_kl_loss)

        return {
            'train_loss': loss,
            'reconstruction_loss': ave_recon_loss,
            'kl_loss': ave_kl_loss,
            'elbo': -(ave_recon_loss + ave_kl_loss),
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

def estimate_shared(z0_mean, z0_logvar, z1_mean, z1_logvar):
    """
    Core of the adaptive VAE algorithm, estimating which factors
    have changed (or in this case which are shared and should remained unchanged
    by being be averaged) between pairs of observations.
    """
    # shared elements that need to be averaged, computed per pair in the batch.
    kl_deltas = kl_normal_loss_pair_elements(z0_mean, z0_logvar, z1_mean, z1_logvar)  # [𝛿_i ...]
    kl_threshs = estimate_kl_threshold(kl_deltas)  # threshold τ

    shared_mask = kl_deltas < kl_threshs  # true if 'unchanged' and should be average

    return kl_deltas, kl_threshs, shared_mask


# ========================================================================= #
# Averaging Functions                                                       #
# ========================================================================= #


def compute_average_gvae(z0_mean, z0_logvar, z1_mean, z1_logvar):
    """
    Compute the arithmetic mean of the encoder distributions.
    - Ada-GVAE Averaging function
    """
    # helper
    z0_var, z1_var = z0_logvar.exp(), z1_logvar.exp()

    # averages
    ave_var = (z0_var + z1_var) * 0.5
    ave_mean = (z0_mean + z1_mean) * 0.5

    # mean, logvar
    return ave_mean, ave_var.log()  # natural log

def compute_average_ml_vae(z0_mean, z0_logvar, z1_mean, z1_logvar):
    """
    Compute the product of the encoder distributions.
    - Ada-ML-VAE Averaging function
    """
    # helper
    z0_var, z1_var = z0_logvar.exp(), z1_logvar.exp()

    # Diagonal matrix inverse: E^-1 = 1 / E
    # https://proofwiki.org/wiki/Inverse_of_Diagonal_Matrix
    z0_invvar, z1_invvar = z0_var.reciprocal(), z1_var.reciprocal()

    # average var: E^-1 = E1^-1 + E2^-1
    ave_var = (z0_invvar + z1_invvar).reciprocal()

    # average mean: u^T = (u1^T E1^-1 + u2^T E2^-1) E
    # u^T is horr vec (u is vert). E is square matrix
    ave_mean = (z0_mean * z0_invvar + z1_mean * z1_invvar) * ave_var

    # mean, logvar
    return ave_mean, ave_var.log()  # natural log


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
