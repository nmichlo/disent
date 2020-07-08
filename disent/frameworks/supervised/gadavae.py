from disent.frameworks.semisupervised.adavae import (AdaVaeLoss, InterceptZMixin, estimate_unchanged)
from disent.frameworks.unsupervised.vae import bce_loss, kl_normal_loss

# ========================================================================= #
# Ada-GVAE                                                                  #
# ========================================================================= #


# def compute_average_gvae_triplet(z_mean, z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
#     """
#     Compute the arithmetic mean of the encoder distributions.
#     - Ada-GVAE Averaging function
#     """
#     # helper
#     z_var, p_z_var, n_z_var = z_logvar.exp(), p_z_logvar.exp(), n_z_logvar.exp()
#
#     # averages
#     ave_var = (z_var + p_z_var + n_z_var) * (1/3)
#     ave_mean = (z_mean + p_z_mean + n_z_mean) * (1/3)
#
#     # mean, logvar
#     return ave_mean, ave_var.log()  # natural log

def compute_constrained_mask(p_kl_deltas, p_unchanged_mask, n_unchanged_mask):
    batch_size, dims = p_kl_deltas.shape

    # number of changed factors
    p_k = torch.sum(~p_unchanged_mask, dim=1, keepdim=True)
    n_k = torch.sum(~n_unchanged_mask, dim=1, keepdim=True)

    sort_indices = torch.argsort(p_kl_deltas, dim=1)

    # orig_indices = torch.arange(dims).repeat(batch_size, 1)
    # orig_mask = orig_indices < k

    # TODO: this is inefficient
    p_mask = torch.zeros_like(p_unchanged_mask)
    for i, min_k in enumerate(torch.min(p_k, n_k)):
        unchanged = dims - min_k
        p_mask[i, sort_indices[i, :unchanged]] = True

    return p_mask


class GuidedAdaVaeLoss(AdaVaeLoss, InterceptZMixin):

    def __init__(self, beta=4, average_mode='gvae'):
        assert average_mode == 'gvae', f'{self.__class__.__name__} currently only supports GVAE averaging (average_mode="gvae")'
        super().__init__(beta=beta, average_mode=average_mode)

        # TODO: remove, this is debug stuff
        self.count = 0

    @property
    def required_observations(self):
        return 3

    def intercept_z(self, a_z_mean, a_z_logvar, *args, **kwargs):
        """
        *NB* arguments must satisfy: d(l, l2) < d(l, l3) [positive dist < negative dist]
        - This function assumes that the distance between labels l, l2, l3
          corresponding to z, z2, z3 satisfy the criteria d(l, l2) < d(l, l3)
          ie. l2 is the positive sample, l3 is the negative sample
        """
        p_z_mean, p_z_logvar, n_z_mean, n_z_logvar = args
        assert not kwargs

        # shared elements that need to be averaged, computed per pair in the batch.
        p_kl_deltas, p_kl_threshs, p_ave_mask = estimate_unchanged(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar)
        n_kl_deltas, n_kl_threshs, n_ave_mask = estimate_unchanged(a_z_mean, a_z_logvar, n_z_mean, n_z_logvar)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        p_ave_mask = compute_constrained_mask(p_kl_deltas, (old_p_ave_mask := p_ave_mask), n_ave_mask)

        # TODO: remove, this is debug stuff
        self.count += int(torch.sum(p_ave_mask == old_p_ave_mask).cpu().detach())

        pAz_mean, pAz_logvar, p_z_mean, p_z_logvar = self.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), p_z_mean, p_z_logvar, p_ave_mask)
        nAz_mean, nAz_logvar, n_z_mean, n_z_logvar = self.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), n_z_mean, n_z_logvar, n_ave_mask)
        a_z_mean, a_z_logvar = self.compute_average(pAz_mean, pAz_logvar, nAz_mean, nAz_logvar)

        return a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar

    def compute_loss(self, x, x_recon, z_mean, z_logvar, z_sampled, *args, **kwargs):
        x2, x2_recon, z2_mean, z2_logvar, z2_sampled, x3, x3_recon, z3_mean, z3_logvar, z3_sampled = args

        # reconstruction error & KL divergence losses
        recon_loss = bce_loss(x, x_recon)              # E[log p(x|z)]
        recon2_loss = bce_loss(x2, x2_recon)           # E[log p(x|z)]
        recon3_loss = bce_loss(x2, x2_recon)           # E[log p(x|z)]

        kl_loss = kl_normal_loss(z_mean, z_logvar)     # D_kl(q(z|x) || p(z|x))
        kl2_loss = kl_normal_loss(z2_mean, z2_logvar)  # D_kl(q(z|x) || p(z|x))
        kl3_loss = kl_normal_loss(z2_mean, z2_logvar)  # D_kl(q(z|x) || p(z|x))

        # compute combined loss
        # reduces down to summing the two BetaVAE losses
        loss = (recon_loss + recon2_loss + recon3_loss) + self.beta * (kl_loss + kl2_loss + kl3_loss)
        loss /= 3

        return {
            'loss': loss
            # TODO: 'reconstruction_loss': recon_loss,
            # TODO: 'kl_loss': kl_loss,
            # TODO: 'elbo': -(recon_loss + kl_loss),
        }

# ========================================================================= #
# END                                                                       #
# ========================================================================= #



import numpy
import torch
#
# def estimate_kl_threshold(kl_deltas):
#     """
#     Compute the threshold for each image pair in a batch of kl divergences of all elements of the latent distributions.
#     It should be noted that for a perfectly trained model, this threshold is always correct.
#     """
#
#     # Must return threshold for each image pair, not over entire batch.
#     # specifying the axis returns a tuple of values and indices... better way?
#     threshs = 0.5 * (kl_deltas.max(axis=1, keepdim=True).values + kl_deltas.min(axis=1, keepdim=True).values)
#     return threshs  # re-add the flattened dimension, shape=(batch_size, 1)


# batch_size = 4
# dims = 5
#
# # TEST
# p_kl_deltas = torch.randn((batch_size, dims)) * 10
# n_kl_deltas = torch.randn((batch_size, dims)) * 10
#
# # ACTUAL
# p_kl_threshs = estimate_kl_threshold(p_kl_deltas)
# n_kl_threshs = estimate_kl_threshold(n_kl_deltas)
#
# p_ave_mask = p_kl_deltas < p_kl_threshs
# n_ave_mask = n_kl_deltas < n_kl_threshs

# def compute_constrained_mask(p_kl_deltas, p_ave_mask, n_ave_mask):
#     batch_size, dims = p_kl_deltas.shape
#
#     p_k = torch.sum(~p_ave_mask, dim=1, keepdim=True)
#     n_k = torch.sum(~n_ave_mask, dim=1, keepdim=True)
#
#     # orig_indices = torch.arange(dims).repeat(batch_size, 1)
#     # orig_mask = orig_indices < k
#
#     sort_indices = torch.argsort(p_kl_deltas, dim=1)
#
#     mask = torch.zeros_like(p_ave_mask)
#     for i, min_k in enumerate(torch.min(p_k, n_k)):
#         unchanged = dims - min_k
#         mask[i, sort_indices[i, :unchanged]] = True
#
#     return p_k




# sorted_deltas = kl_deltas.gather(1, sort_indices)

# print(k)

# orig_indices = torch.arange(dims).repeat(batch_size, 1)
# orig_k_masks = (orig_indices >= (k - 1))

# print(orig_indices)
# print(orig_k_masks)




