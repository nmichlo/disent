import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
import numpy as np

# ========================================================================= #
# Interpolate                                                               #
# ========================================================================= #


def _lerp(a, b, t):
    """Linear interpolation between parameters, respects bounds when t is out of bounds [0, 1]"""
    assert a < b
    t = max(0, min(t, 1))
    # precise method, guarantees v==b when t==1 | simplifies to: a + t*(b-a)
    return (1-t)*a + t*b

def _lerp_step(a, b, step, max_steps):
    """Linear interpolation based on a step count."""
    if max_steps <= 0:
        return b
    return _lerp(a, b, step / max_steps)


# ========================================================================= #
# Base Loss                                                                 #
# ========================================================================= #


class BaseLoss(ABC):
    def __call__(self, x, x_recon, z_mean, z_logvar, is_train=True):
        return self.compute_loss(x, x_recon, z_mean, z_logvar, is_train)

    @abstractmethod
    def compute_loss(self, x, x_recon, z_mean, z_logvar, is_train):
        pass


# ========================================================================= #
# Vae Loss                                                                  #
# ========================================================================= #

def _bce_loss_with_logits(x, x_recon, activation='sigmoid'):
    """
    Computes the Bernoulli loss.
    FROM: https://github.com/google-research/disentanglement_lib/blob/76f41e39cdeff8517f7fba9d57b09f35703efca9/disentanglement_lib/methods/shared/losses.py
    """
    x_recon = x_recon.view(x.shape[0], -1)
    x = x.view(x.shape[0], -1)

    if activation == 'sigmoid':
        per_sample_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='none').sum(axis=1)  # tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recon, labels=x), axis=1)
        reconstruction_loss = per_sample_loss.mean()                                                    # tf.reduce_mean(per_sample_loss)
        # SIMPLIFIED: reconstruction_loss = F.binary_cross_entropy_with_logits(x_recon, x, reduction='sum') / x.shape[0]
        # SIMPLIFIED: reconstruction_loss = F.binary_cross_entropy(x_recon, x, reduction='sum') / x.shape[0]
    else:
        raise KeyError(f'Unknown activation: {activation}')

    return reconstruction_loss

def _kl_normal_loss(mu, logvar):
    """
    Calculates the KL divergence between a normal distribution with
    diagonal covariance and a unit normal distribution.
    https://github.com/Schlumberger/joint-vae/blob/master/jointvae/training.py

    Parameters
    ----------
    mu : torch.Tensor
        Mean of the normal distribution. Shape (N, D) where D is dimension
        of distribution.
    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (N, D)
    """
    # Calculate KL divergence
    kl_values = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    # Mean KL divergence across batch for each latent variable
    kl_means = torch.mean(kl_values, dim=0)
    # KL loss is sum of mean KL of each latent variable
    kl_loss = torch.sum(kl_means)

    return kl_loss


class VaeLoss(BaseLoss):
    def compute_loss(self, x, x_recon, z_mean, z_logvar, is_train):
        # reconstruction error & KL divergence losses
        recon_loss = _bce_loss_with_logits(x, x_recon) # E[log p(x|z)]
        kl_loss = _kl_normal_loss(z_mean, z_logvar)      # D_kl(q(z|x) || p(z|x))
        # compute combined loss
        return recon_loss + kl_loss

        # DISENTANGLEMENT LIB:
        # per_sample_loss = losses.make_reconstruction_loss(features, reconstructions) # [bernoulli_loss with GIN config]
        # reconstruction_loss = tf.reduce_mean(per_sample_loss)

        # kl_loss = compute_gaussian_kl(z_mean, z_logvar)
        # regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)

        # loss = tf.add(reconstruction_loss, regularizer, name="loss")
        # elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")




# https://github.com/google-research/disentanglement_lib/blob/a64b8b9994a28fafd47ccd866b0318fa30a3c76c/disentanglement_lib/methods/unsupervised/vae.py#L153
# class BaseVAE(gaussian_encoder_model.GaussianEncoderModel):
#   """Abstract base class of a basic Gaussian encoder model."""
#
#   def model_fn(self, features, labels, mode, params):
#     """TPUEstimator compatible model function."""
#     del labels
#     is_training = (mode == tf.estimator.ModeKeys.TRAIN)
#     data_shape = features.get_shape().as_list()[1:]
#     z_mean, z_logvar = self.gaussian_encoder(features, is_training=is_training)
#     z_sampled = self.sample_from_latent_distribution(z_mean, z_logvar)
#     reconstructions = self.decode(z_sampled, data_shape, is_training)
#     per_sample_loss = losses.make_reconstruction_loss(features, reconstructions)
#     reconstruction_loss = tf.reduce_mean(per_sample_loss)
#     kl_loss = compute_gaussian_kl(z_mean, z_logvar)
#     regularizer = self.regularizer(kl_loss, z_mean, z_logvar, z_sampled)
#     loss = tf.add(reconstruction_loss, regularizer, name="loss")
#     elbo = tf.add(reconstruction_loss, kl_loss, name="elbo")
#     if mode == tf.estimator.ModeKeys.TRAIN:
#       optimizer = optimizers.make_vae_optimizer()
#       update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#       train_op = optimizer.minimize(
#           loss=loss, global_step=tf.train.get_global_step())
#       train_op = tf.group([train_op, update_ops])
#       tf.summary.scalar("reconstruction_loss", reconstruction_loss)
#       tf.summary.scalar("elbo", -elbo)
#
#       logging_hook = tf.train.LoggingTensorHook({
#           "loss": loss,
#           "reconstruction_loss": reconstruction_loss,
#           "elbo": -elbo
#       },
#                                                 every_n_iter=100)
#       return tf.contrib.tpu.TPUEstimatorSpec(
#           mode=mode,
#           loss=loss,
#           train_op=train_op,
#           training_hooks=[logging_hook])
#     elif mode == tf.estimator.ModeKeys.EVAL:
#       return tf.contrib.tpu.TPUEstimatorSpec(
#           mode=mode,
#           loss=loss,
#           eval_metrics=(make_metric_fn("reconstruction_loss", "elbo",
#                                        "regularizer", "kl_loss"),
#                         [reconstruction_loss, -elbo, regularizer, kl_loss]))
#     else:
#       raise NotImplementedError("Eval mode not supported.")
#
#   def gaussian_encoder(self, input_tensor, is_training):
#     """Applies the Gaussian encoder to images.
#     Args:
#       input_tensor: Tensor with the observations to be encoded.
#       is_training: Boolean indicating whether in training mode.
#     Returns:
#       Tuple of tensors with the mean and log variance of the Gaussian encoder.
#     """
#     return architectures.make_gaussian_encoder(
#         input_tensor, is_training=is_training)
#
#   def decode(self, latent_tensor, observation_shape, is_training):
#     """Decodes the latent_tensor to an observation."""
#     return architectures.make_decoder(
#         latent_tensor, observation_shape, is_training=is_training)


# ========================================================================= #
# Beta-VAE Loss                                                             #
# ========================================================================= #

class BetaVaeLoss(VaeLoss):
    def __init__(self, beta=4):
        self.beta = beta

    def compute_loss(self, x, x_recon, z_mean, z_logvar, is_train):
        # reconstruction error & KL divergence losses
        recon_loss = _bce_loss_with_logits(x, x_recon)  # E[log p(x|z)]
        kl_loss = _kl_normal_loss(z_mean, z_logvar)       # D_kl(q(z|x) || p(z|x))
        # compute combined loss
        return recon_loss + self.beta * kl_loss


# class BetaVaeHLoss(BetaVaeLoss):
#     """
#     Compute the Beta-VAE loss as in [1]
#
#     [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
#     a constrained variational framework." (2016).
#     """
#
#     def __init__(self, beta=4, anneal_end_steps=None):
#         super().__init__(beta)
#         self.n_train_steps = 0
#         self.anneal_end_steps = anneal_end_steps  # TODO
#
#     def compute_loss(self, x, x_recon, z_mean, z_logvar, is_train):
#         # reconstruction error & KL divergence losses
#         recon_loss = _bce_loss(x, x_recon)  # E[log p(x|z)]
#         kl_loss = _kl_normal_loss(z_mean, z_logvar)  # D_kl(q(z|x) || p(z|x))
#         # increase beta over time
#         anneal_reg = _lerp_step(0, 1, self.n_train_steps, self.anneal_end_steps) if is_train else 1
#         # compute combined loss
#         return recon_loss + (anneal_reg * self.beta) * kl_loss


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
