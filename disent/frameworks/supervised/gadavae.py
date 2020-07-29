from disent.frameworks.semisupervised.adavae import (AdaVaeLoss, estimate_shared)

import numpy as np
import torch
import logging

from disent.frameworks.unsupervised.vae import bce_loss_with_logits, kl_normal_loss

log = logging.getLogger(__name__)


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


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


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class GuidedAdaVaeLoss(AdaVaeLoss):

    MODE_ADAVAE = 'adavae'
    MODE_AVE_POS = 'ave_pos'
    MODE_AVE_TRIPLET = 'ave_triple'

    def __init__(
            self,
            beta=4,
            average_mode=AdaVaeLoss.AVE_MODE_GVAE,
            mode=MODE_AVE_TRIPLET,
            triplet_scale=0,
            triplet_alpha=0.3,
            triplet_after_sampling=False,
    ):
        assert average_mode == AdaVaeLoss.AVE_MODE_GVAE, f'currently only supports average_mode={repr(AdaVaeLoss.AVE_MODE_GVAE)}'
        super().__init__(beta=beta, average_mode=average_mode)

        # set mode
        assert mode in {GuidedAdaVaeLoss.MODE_ADAVAE, GuidedAdaVaeLoss.MODE_AVE_POS, GuidedAdaVaeLoss.MODE_AVE_TRIPLET}, f'invalid {mode=}, must be one of {[GuidedAdaVaeLoss.MODE_ADAVAE, GuidedAdaVaeLoss.MODE_AVE_POS, GuidedAdaVaeLoss.MODE_AVE_TRIPLET]}'
        self.mode = mode

        # use triplet loss
        self.triplet_scale = triplet_scale
        self.triplet_alpha = triplet_alpha
        self.triplet_after_sampling = triplet_after_sampling
        assert self.triplet_scale >= 0, f'triplet_scale={repr(self.triplet_scale)} must be non-negative'
        if self.triplet_scale > 0:
            assert mode == GuidedAdaVaeLoss.MODE_AVE_TRIPLET, f'triplet_scale={repr(self.triplet_scale)}, only supports triplet_scale > 0 for mode={repr(GuidedAdaVaeLoss.MODE_AVE_TRIPLET)}'

        # DEBUG VARS
        self.p_shared_before = 0
        self.p_shared_after = 0
        self.n_shared_before = 0
        self.n_shared_after = 0
        self.item_count = 0

    @property
    def required_observations(self):
        return 3

    def intercept_z(self, z_params, *args):
        """
        *NB* arguments must satisfy: d(l, l2) < d(l, l3) [positive dist < negative dist]
        - This function assumes that the distance between labels l, l2, l3
          corresponding to z, z2, z3 satisfy the criteria d(l, l2) < d(l, l3)
          ie. l2 is the positive sample, l3 is the negative sample
        """
        [(a_z_mean, a_z_logvar), (p_z_mean, p_z_logvar), (n_z_mean, n_z_logvar)] = (z_params, *args)

        # shared elements that need to be averaged, computed per pair in the batch.
        p_kl_deltas, p_kl_threshs, p_shared_mask = estimate_shared(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar)
        n_kl_deltas, n_kl_threshs, n_shared_mask = estimate_shared(a_z_mean, a_z_logvar, n_z_mean, n_z_logvar)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        old_p_ave_mask, old_n_ave_mask = p_shared_mask, n_shared_mask
        p_shared_mask = compute_constrained_mask(p_kl_deltas, old_p_ave_mask, n_kl_deltas, old_n_ave_mask, positive=True)
        n_shared_mask = compute_constrained_mask(p_kl_deltas, old_p_ave_mask, n_kl_deltas, old_n_ave_mask, positive=False)

        # DEBUG
        self.item_count += len(a_z_mean)
        self.p_shared_before += int(old_p_ave_mask.sum())
        self.p_shared_after += int(p_shared_mask.sum())
        self.n_shared_before += int(old_n_ave_mask.sum())
        self.n_shared_after += int(n_shared_mask.sum())

        if self.mode == 'adavae':
            return super().intercept_z((a_z_mean, a_z_logvar), (p_z_mean, p_z_logvar))
        elif self.mode == 'ave_pos':
            return self.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), p_z_mean, p_z_logvar, p_shared_mask)
        elif self.mode == 'ave_triple':
            (pAz_mean, pAz_logvar), (p_z_mean, p_z_logvar) = self.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), p_z_mean, p_z_logvar, p_shared_mask)
            (nAz_mean, nAz_logvar), (n_z_mean, n_z_logvar) = self.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), n_z_mean, n_z_logvar, n_shared_mask)
            a_z_mean, a_z_logvar = self.compute_average(pAz_mean, pAz_logvar, nAz_mean, nAz_logvar)
            return (a_z_mean, a_z_logvar), (p_z_mean, p_z_logvar), (n_z_mean, n_z_logvar)
        else:
            raise KeyError

    def compute_loss(self, forward_data, *args):
        # COMPUTE LOSS FOR TRIPLE:
        if self.mode == 'ave_triple':
            [(x, x_recon, (z_mean, z_logvar), z_sampled),
             (x2, x2_recon, (z2_mean, z2_logvar), z2_sampled),
             (x3, x3_recon, (z3_mean, z3_logvar), z3_sampled)] = (forward_data, *args)

            # reconstruction error & KL divergence losses
            recon_loss = bce_loss_with_logits(x, x_recon)     # E[log p(x|z)]
            recon2_loss = bce_loss_with_logits(x2, x2_recon)  # E[log p(x|z)]
            recon3_loss = bce_loss_with_logits(x3, x3_recon)  # E[log p(x|z)]
            ave_recon_loss = (recon_loss + recon2_loss + recon3_loss) / 3

            kl_loss = kl_normal_loss(z_mean, z_logvar)     # D_kl(q(z|x) || p(z|x))
            kl2_loss = kl_normal_loss(z2_mean, z2_logvar)  # D_kl(q(z|x) || p(z|x))
            kl3_loss = kl_normal_loss(z3_mean, z3_logvar)  # D_kl(q(z|x) || p(z|x))
            ave_kl_loss = (kl_loss + kl2_loss + kl3_loss) / 3

            # compute combined loss
            # reduces down to summing the two BetaVAE losses
            loss = (recon_loss + recon2_loss + recon3_loss) + self.beta * (kl_loss + kl2_loss + kl3_loss)
            loss /= 3

            loss_dict = {
                'train_loss': loss,
                'reconstruction_loss': ave_recon_loss,
                'kl_loss': ave_kl_loss,
                'elbo': -(ave_recon_loss + ave_kl_loss),
            }

            if self.triplet_scale > 0:
                if self.triplet_after_sampling:
                    loss_triplet = triplet_loss(z_sampled, z2_sampled, z3_sampled, alpha=self.triplet_alpha)
                else:
                    loss_triplet = triplet_loss(z_mean, z2_mean, z3_mean, alpha=self.triplet_alpha)
                loss_dict.update({
                    'train_loss': loss + self.triplet_scale * loss_triplet,
                    'triplet_loss': loss_triplet
                })

        # COMPUTE LOSS FOR PAIR:
        elif (self.mode == 'adavae') or (self.mode == 'ave_pos'):
            assert self.triplet_scale == 0, f'triplet_scale={repr(self.triplet_scale)}, triplet_scale > 0 is not supported for the current mode={self.mode}'
            loss_dict = super().compute_loss(forward_data, *args)
        else:
            raise KeyError

        # DEBUG COUNTS
        p_shared_before = self.p_shared_before / self.item_count
        p_shared_after = self.p_shared_after / self.item_count
        n_shared_before = self.n_shared_before / self.item_count
        n_shared_after = self.n_shared_after / self.item_count
        self.p_shared_before, self.p_shared_after, self.n_shared_before, self.n_shared_after = 0, 0, 0, 0

        loss_dict.update({
            'p_shared_before': p_shared_before,
            'p_shared_after': p_shared_after,
            'n_shared_before': n_shared_before,
            'n_shared_after': n_shared_after,
        })

        return loss_dict


# ========================================================================= #
# TRIPLET LOSSES                                                            #
# ========================================================================= #

def triplet_loss(anchor, positive, negative, alpha=0.3):
    # import tensorflow as tf
    # positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
    # negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
    # loss_1 = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
    # loss = tf.reduce_sum(tf.maximum(loss_1, 0.0), 0)

    positive_dist = torch.sum((anchor - positive)**2, dim=1)
    negative_dist = torch.sum((anchor - negative)**2, dim=1)
    clamped = torch.clamp_min((positive_dist - negative_dist) + alpha, 0)
    loss = torch.sum(clamped, dim=0)
    return loss


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

