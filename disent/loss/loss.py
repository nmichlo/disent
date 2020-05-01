import torch
import torch.nn.functional as F


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


# def _bce_loss_with_logits(x, x_recon):
#     """
#     Computes the Bernoulli loss for the sigmoid activation function
#     FROM: https://github.com/google-research/disentanglement_lib/blob/76f41e39cdeff8517f7fba9d57b09f35703efca9/disentanglement_lib/methods/shared/losses.py
#     """
#     x, x_recon = x.view(x.shape[0], -1), x_recon.view(x.shape[0], -1)
#     per_sample_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none').sum(axis=1)  # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x), axis=1)
#     reconstruction_loss = per_sample_loss.mean()                                                    # tf.reduce_mean(per_sample_loss)
#     return reconstruction_loss  # F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum') / x.shape[0]

def _bce_loss(x, x_recon):
    """Computes the Bernoulli loss"""
    x, x_recon = x.view(x.shape[0], -1), x_recon.view(x.shape[0], -1)
    per_sample_loss = F.binary_cross_entropy(x_recon, x, reduction='none').sum(axis=1)
    reconstruction_loss = per_sample_loss.mean()
    return reconstruction_loss


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


# ========================================================================= #
# Base VAE Loss                                                             #
# ========================================================================= #


class VaeLoss(object):

    def __call__(self, x, x_recon, z_mean, z_logvar, z_sampled, *args, **kwargs):
        return self.compute_loss(x, x_recon, z_mean, z_logvar, z_sampled, *args, **kwargs)

    @property
    def is_pair_loss(self):
        """override in subclasses that need paired loss, indicates format of arguments needed for compute_loss"""
        return False

    def compute_loss(self, x, x_recon, z_mean, z_logvar, z_sampled, *args, **kwargs):
        """
        Compute the varous VAE loss components.
        Based on: https://github.com/google-research/disentanglement_lib/blob/a64b8b9994a28fafd47ccd866b0318fa30a3c76c/disentanglement_lib/methods/unsupervised/vae.py#L153
        """
        # reconstruction loss
        recon_loss = _bce_loss(x, x_recon)   # E[log p(x|z)]

        # regularizer
        kl_loss = _kl_normal_loss(z_mean, z_logvar)      # D_kl(q(z|x) || p(z|x))
        regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)

        return {
            'loss': recon_loss + regularizer,
            # TODO: 'reconstruction_loss': recon_loss,
            # TODO: 'kl_loss': kl_loss,
            # TODO: 'elbo': -(recon_loss + kl_loss),
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


def lerp(a, b, t):
    """Linear interpolation between parameters, respects bounds when t is out of bounds [0, 1]"""
    assert a < b
    t = max(0, min(t, 1))
    # precise method, guarantees v==b when t==1 | simplifies to: a + t*(b-a)
    return (1-t)*a + t*b


def lerp_step(a, b, step, max_steps):
    """Linear interpolation based on a step count."""
    if max_steps <= 0:
        return b
    return lerp(a, b, step / max_steps)


class BetaVaeHLoss(BetaVaeLoss):
    """
    Compute the Beta-VAE loss as in [1]

    [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
    a constrained variational framework." (2016).
    """

    def __init__(self, anneal_end_steps, beta=4):
        super().__init__(beta)
        self.n_train_steps = 0
        self.anneal_end_steps = anneal_end_steps
        raise NotImplementedError('n_train_steps is not yet implemented for BetaVaeHLoss, it will not yet work')

    def regularizer(self, kl_loss, z_mean, z_logvar, z_sampled):
        print('WARNING: training steps not updated')
        anneal_reg = lerp_step(0, 1, self.n_train_steps, self.anneal_end_steps) # if is_train else 1
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
    Compute the threshold for each image pair in a batch of kl divergences of all elements of the latent distributions.
    It should be noted that for a perfectly trained model, this threshold is always correct.
    """

    # Must return threshold for each image pair, not over entire batch.
    # specifying the axis returns a tuple of values and indices... better way?
    threshs = 0.5 * (kl_deltas.max(axis=1)[0] + kl_deltas.min(axis=1)[0])
    return threshs[:, None]  # re-add the flattened dimension, shape=(batch_size, 1)

class AdaGVaeLoss(BetaVaeLoss):

    def __init__(self, beta=4):
        super().__init__(beta)
        # self.sampler = sampler
        # self.vae = vae

    @property
    def is_pair_loss(self):
        return True

    def compute_loss(self, x, x_recon, z_mean, z_logvar, z_sampled, *args, **kwargs):
        x2, x2_recon, z2_mean, z2_logvar, z2_sampled = args

        # shared elements that need to be averaged, computed per pair in the batch.
        kl_deltas = _kl_normal_loss_pair_elements(z_mean, z_logvar, z2_mean, z2_logvar)  # [𝛿_i ...]
        kl_threshs = _estimate_kl_threshold(kl_deltas)                                    # threshold τ
        ave_elements = kl_deltas < kl_threshs

        # compute average posteriors
        ave_mu, ave_logvar = self.compute_average(z_mean, z_logvar, z2_mean, z2_logvar)

        # compute approximate posteriors
        # approx_z_mean, approx_z_logvar = z_mean.clone(), z_logvar.clone()
        # approx_z2_mean, approx_z2_logvar = z2_mean.clone(), z2_logvar.clone()
        z_mean[ave_elements], z_logvar[ave_elements] = ave_mu[ave_elements], ave_logvar[ave_elements]
        z2_mean[ave_elements], z2_logvar[ave_elements] = ave_mu[ave_elements], ave_logvar[ave_elements]

        # TODO: x_recon and x2_recon need to use updated/averaged z?
        # TODO: make use of regularizer() function
        # reconstruction error & KL divergence losses
        recon_loss = _bce_loss(x, x_recon)              # E[log p(x|z)]
        recon2_loss = _bce_loss(x2, x2_recon)           # E[log p(x|z)]
        kl_loss = _kl_normal_loss(z_mean, z_logvar)     # D_kl(q(z|x) || p(z|x))
        kl2_loss = _kl_normal_loss(z2_mean, z2_logvar)  # D_kl(q(z|x) || p(z|x))

        # compute combined loss
        loss = (recon_loss + recon2_loss) + self.beta * (kl_loss + kl2_loss)

        return {
            'loss': loss
            # TODO: 'reconstruction_loss': recon_loss,
            # TODO: 'kl_loss': kl_loss,
            # TODO: 'elbo': -(recon_loss + kl_loss),
        }

    @staticmethod
    def compute_average(z_mean, z_logvar, z2_mean, z2_logvar):
        """
        Compute the arithmetic mean for the mean and variance.
        - Ada-GVAE Averaging function
        """
        # helper
        z_var, z2_var = z_logvar.exp(), z2_logvar.exp()

        # averages
        ave_var = (z_var + z2_var) * 0.5
        ave_mean = (z_mean + z2_mean) * 0.5

        # mean, logvar
        return ave_mean, ave_var.log()  # natural log


class AdaMlVaeLoss(AdaGVaeLoss):

    @staticmethod
    def compute_average(z_mean, z_logvar, z2_mean, z2_logvar):
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
        ave_mean = (z_mean*z_invvar + z2_mean*z2_invvar) * ave_var

        # mean, logvar
        return ave_mean, ave_var.log()  # natural log

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
