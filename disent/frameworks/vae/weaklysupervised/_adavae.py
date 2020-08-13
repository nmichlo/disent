from disent.frameworks.vae.unsupervised import BetaVae
from disent.frameworks.vae.loss import bce_loss_with_logits, kl_normal_loss


# ========================================================================= #
# Ada-GVAE                                                                  #
# ========================================================================= #


class AdaVae(BetaVae):

    def __init__(self, make_optimizer_fn, make_model_fn, beta=4, average_mode='gvae'):
        super().__init__(make_optimizer_fn, make_model_fn, beta=beta)
        # averaging modes
        self.compute_average = {
            'gvae': compute_average_gvae,
            'ml-vae': compute_average_ml_vae
        }[average_mode]

    def compute_loss(self, batch, batch_idx):
        x0, x1 = batch
        # FORWARD
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # latent distribution parametrisation
        z0_mean, z0_logvar = self.encode_gaussian(x0)
        z1_mean, z1_logvar = self.encode_gaussian(x1)
        # intercept and mutate z [SPECIFIC TO ADAVAE]
        (z0_mean, z0_logvar, z1_mean, z1_logvar), intercept_logs = self.intercept_z(z0_mean, z0_logvar, z1_mean, z1_logvar)
        # sample from latent distribution
        z0_sampled = self.reparameterize(z0_mean, z0_logvar)
        z1_sampled = self.reparameterize(z1_mean, z1_logvar)
        # reconstruct without the final activation
        x0_recon = self.decode_partial(z0_sampled)
        x1_recon = self.decode_partial(z1_sampled)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        # LOSS
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        # reconstruction error
        recon0_loss = bce_loss_with_logits(x0, x0_recon)  # E[log p(x|z)]
        recon1_loss = bce_loss_with_logits(x1, x1_recon)  # E[log p(x|z)]
        ave_recon_loss = (recon0_loss + recon1_loss) / 2
        # KL divergence
        kl0_loss = kl_normal_loss(z0_mean, z0_logvar)     # D_kl(q(z|x) || p(z|x))
        kl1_loss = kl_normal_loss(z1_mean, z1_logvar)     # D_kl(q(z|x) || p(z|x))
        ave_kl_loss = (kl0_loss + kl1_loss) / 2
        # compute kl regularisation
        ave_kl_reg_loss = self.kl_regularization(ave_kl_loss)
        # compute combined loss - must be same as the BetaVAE
        loss = ave_recon_loss + ave_kl_reg_loss
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

        return {
            'train_loss': loss,
            'recon_loss': ave_recon_loss,
            'kl_reg_loss': ave_kl_reg_loss,
            'kl_loss': ave_kl_loss,
            'elbo': -(ave_recon_loss + ave_kl_loss),
            **intercept_logs,
        }

    def intercept_z(self, z0_mean, z0_logvar, z1_mean, z1_logvar):
        # shared elements that need to be averaged, computed per pair in the batch.
        _, _, share_mask = AdaVae.estimate_shared(z0_mean, z0_logvar, z1_mean, z1_logvar)
        # make averaged z parameters
        new_args = self.make_averaged(z0_mean, z0_logvar, z1_mean, z1_logvar, share_mask)
        # return new args & generate logs
        return new_args, {'shared': share_mask.sum(dim=1).float().mean()}

    def make_averaged(self, z0_mean, z0_logvar, z1_mean, z1_logvar, share_mask):
        # compute average posteriors
        ave_mu, ave_logvar = self.compute_average(z0_mean, z0_logvar, z1_mean, z1_logvar)
        # apply average
        z0_mean = (~share_mask * z0_mean) + (share_mask * ave_mu)
        z1_mean = (~share_mask * z1_mean) + (share_mask * ave_mu)
        z0_logvar = (~share_mask * z0_logvar) + (share_mask * ave_logvar)
        z1_logvar = (~share_mask * z1_logvar) + (share_mask * ave_logvar)
        # return values
        return z0_mean, z0_logvar, z1_mean, z1_logvar

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @staticmethod
    def estimate_shared(z0_mean, z0_logvar, z1_mean, z1_logvar):
        """
        Core of the adaptive VAE algorithm, estimating which factors
        have changed (or in this case which are shared and should remained unchanged
        by being be averaged) between pairs of observations.
        """
        # shared elements that need to be averaged, computed per pair in the batch.
        # [𝛿_i ...]
        kl_deltas = kl_normal_loss_pair_elements(z0_mean, z0_logvar, z1_mean, z1_logvar)
        # threshold τ
        kl_threshs = estimate_kl_threshold(kl_deltas)
        # check shapes
        assert kl_threshs.shape == (z0_mean.shape[0], 1), f'{kl_threshs.shape} != {(z0_mean.shape[0], 1)}'
        # true if 'unchanged' and should be average
        shared_mask = kl_deltas < kl_threshs
        # return
        return kl_deltas, kl_threshs, shared_mask


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def kl_normal_loss_pair_elements(z0_mean, z0_logvar, z1_mean, z1_logvar):
    """Compute the KL divergence for normal distributions between all corresponding elements of a pair of latent vectors"""
    # compute GVAE deltas
    # σ0 = logv0.exp() ** 0.5
    # σ1 = logv1.exp() ** 0.5
    # return 0.5 * ((σ0/σ1)**2 + ((μ1 - μ0)**2)/(σ1**2) - 1 + 2*ln(σ1/σ0))
    # return 0.5 * ((σ0/σ1)**2 + ((μ1 - μ0)**2)/(σ1**2) - 1 + ln(σ1**2 / σ0**2))
    # return 0.5 * (σ0.exp()/σ1.exp() + (μ1 - μ0).pow(2)/σ1.exp() - 1 + (logv1 - logv0))
    return 0.5 * ((z0_logvar.exp() / z1_logvar.exp()) + (z1_mean - z0_mean).pow(2) / z1_logvar.exp() - 1 + (z1_logvar - z0_logvar))

def estimate_kl_threshold(kl_deltas):
    """
    Compute the threshold for each image pair in a batch of kl divergences of all elements of the latent distributions.
    It should be noted that for a perfectly trained model, this threshold is always correct.
    """
    # TODO: what would happen if you took the std across the batch and the std across the vector
    #       and then took one less than the other for the thresh? What is that intuition?
    # TODO: what would happen if you used a ratio between min and max instead of the mask and hard averaging
    maximums = kl_deltas.max(axis=1, keepdim=True).values
    minimums = kl_deltas.min(axis=1, keepdim=True).values
    return 0.5 * (minimums + maximums)


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
