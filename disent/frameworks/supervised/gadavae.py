from disent.frameworks.semisupervised.adavae import (AdaVaeLoss, estimate_shared)

import numpy as np
import torch
import logging

log = logging.getLogger(__name__)

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

def compute_constrained_mask(p_kl_deltas, p_shared_mask, n_kl_deltas, n_shared_mask, positive=True):
    batch_size, dims = p_kl_deltas.shape

    # number of changed factors
    p_shared = torch.sum(p_shared_mask, dim=1, keepdim=True)
    n_shared = torch.sum(n_shared_mask, dim=1, keepdim=True)

    # order from smallest to largest
    sort_indices = torch.argsort(p_kl_deltas if positive else n_kl_deltas, dim=1)
    new_mask = torch.zeros_like(p_shared_mask)

    # share max
    new_share = torch.max(p_shared, n_shared) if positive else torch.min(p_shared, n_shared)
    assert new_share.shape == p_shared.shape == n_shared.shape

    # compute p_k should be less than n_k
    for i, shared in enumerate(new_share):
        new_mask[i, sort_indices[i, :shared]] = True

    return new_mask

class GuidedAdaVaeLoss(AdaVaeLoss):

    def __init__(self, beta=4, average_mode='gvae'):
        assert average_mode == 'gvae', f'{self.__class__.__name__} currently only supports GVAE averaging (average_mode="gvae")'
        super().__init__(beta=beta, average_mode=average_mode)

        # TODO: remove, this is debug stuff
        self.p_count = 0
        self.p_count_new = 0
        self.n_count = 0
        self.n_count_new = 0
        self.p_gt_count = 0
        self.n_gt_count = 0
        self.item_count = 0

    @property
    def required_observations(self):
        return 3

    # def intercept_z(self, a_z_mean, a_z_logvar, *args, **kwargs):
    #     p_z_mean, p_z_logvar, n_z_mean, n_z_logvar = args
    #     assert not kwargs
    #
    #     # shared elements that need to be averaged, computed per pair in the batch.
    #     p_kl_deltas, p_kl_threshs, p_ave_mask = estimate_shared(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar)
    #     n_kl_deltas, n_kl_threshs, n_ave_mask = estimate_shared(a_z_mean, a_z_logvar, n_z_mean, n_z_logvar)
    #
    #     old_p_ave_mask, old_n_ave_mask = p_ave_mask, n_ave_mask
    #     p_ave_mask = compute_constrained_mask(p_kl_deltas, old_p_ave_mask, n_kl_deltas, old_n_ave_mask, positive=True)
    #
    #     # DEBUG
    #     # self.p_count += int(torch.sum(old_p_ave_mask))
    #     # self.p_count_new += int(torch.sum(p_ave_mask))
    #     # self.n_count += int(torch.sum(n_ave_mask))
    #
    #     # make averaged z parameters
    #     return self.make_averaged(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, p_ave_mask)

    def intercept_z(self, z_params, *args):
        """
        *NB* arguments must satisfy: d(l, l2) < d(l, l3) [positive dist < negative dist]
        - This function assumes that the distance between labels l, l2, l3
          corresponding to z, z2, z3 satisfy the criteria d(l, l2) < d(l, l3)
          ie. l2 is the positive sample, l3 is the negative sample
        """
        [(a_z_mean, a_z_logvar), (p_z_mean, p_z_logvar), (n_z_mean, n_z_logvar)] = (z_params, *args)

        debug = False
        a_factors, p_factors, n_factors = (None, None, None)

        # shared elements that need to be averaged, computed per pair in the batch.
        # TODO: WHY IS THIS NOT WORKING? WHY IS SUM(p_ave_mask) < SUM(n_ave_mask)
        p_kl_deltas, p_kl_threshs, p_ave_mask = estimate_shared(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar)
        n_kl_deltas, n_kl_threshs, n_ave_mask = estimate_shared(a_z_mean, a_z_logvar, n_z_mean, n_z_logvar)

        if debug and a_factors:
            p_ground_truth_shared = np.around(float((a_factors == p_factors).sum(axis=1).float().mean()), 1)
            p_mask_shared = np.around(float(p_ave_mask.sum(axis=1).float().mean()), 1)
            n_ground_truth_shared = np.around(float((a_factors == n_factors).sum(axis=1).float().mean()), 1)
            n_mask_shared = np.around(float(n_ave_mask.sum(axis=1).float().mean()), 1)
            log.debug(p_ground_truth_shared, '<', n_ground_truth_shared, '|', p_mask_shared, '<', n_mask_shared)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        old_p_ave_mask, old_n_ave_mask = p_ave_mask, n_ave_mask
        p_ave_mask = compute_constrained_mask(p_kl_deltas, old_p_ave_mask, n_kl_deltas, old_n_ave_mask, positive=True)
        n_ave_mask = compute_constrained_mask(p_kl_deltas, old_p_ave_mask, n_kl_deltas, old_n_ave_mask, positive=False)

        # DEBUG
        # TODO: WHY IS p_count < n_count ?????????????
        self.item_count += len(a_z_mean)
        self.p_count += int(old_p_ave_mask.sum())
        self.p_count_new += int(p_ave_mask.sum())
        self.n_count += int(old_n_ave_mask.sum())
        self.n_count_new += int(n_ave_mask.sum())

        if a_factors:
            self.p_gt_count += int((a_factors == p_factors).sum())
            self.n_gt_count += int((a_factors == n_factors).sum())

        return super().intercept_z((a_z_mean, a_z_logvar), (p_z_mean, p_z_logvar))

        # return self.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), p_z_mean, p_z_logvar, p_ave_mask)

        # pAz_mean, pAz_logvar, p_z_mean, p_z_logvar = self.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), p_z_mean, p_z_logvar, p_ave_mask)
        # nAz_mean, nAz_logvar, n_z_mean, n_z_logvar = self.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), n_z_mean, n_z_logvar, n_ave_mask)
        # a_z_mean, a_z_logvar = self.compute_average(pAz_mean, pAz_logvar, nAz_mean, nAz_logvar)
        #
        # return a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar

    # def compute_loss(self, x, x_recon, z_mean, z_logvar, z_sampled, *args, **kwargs):
    #     x2, x2_recon, z2_mean, z2_logvar, z2_sampled, x3, x3_recon, z3_mean, z3_logvar, z3_sampled = args
    #
    #     # reconstruction error & KL divergence losses
    #     recon_loss = bce_loss_with_logits(x, x_recon)     # E[log p(x|z)]
    #     recon2_loss = bce_loss_with_logits(x2, x2_recon)  # E[log p(x|z)]
    #     recon3_loss = bce_loss_with_logits(x2, x2_recon)  # E[log p(x|z)]
    #
    #     kl_loss = kl_normal_loss(z_mean, z_logvar)     # D_kl(q(z|x) || p(z|x))
    #     kl2_loss = kl_normal_loss(z2_mean, z2_logvar)  # D_kl(q(z|x) || p(z|x))
    #     kl3_loss = kl_normal_loss(z2_mean, z2_logvar)  # D_kl(q(z|x) || p(z|x))
    #
    #     # compute combined loss
    #     # reduces down to summing the two BetaVAE losses
    #     loss = (recon_loss + recon2_loss + recon3_loss) + self.beta * (kl_loss + kl2_loss + kl3_loss)
    #     loss /= 3
    #
    #     return {
    #         'loss': loss
    #         # TODO: 'reconstruction_loss': recon_loss,
    #         # TODO: 'kl_loss': kl_loss,
    #         # TODO: 'elbo': -(recon_loss + kl_loss),
    #     }

# ========================================================================= #
# END                                                                       #
# ========================================================================= #

