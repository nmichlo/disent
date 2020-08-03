import gin
import torch

from disent.frameworks.framework import BaseFramework
from disent.frameworks.weaklysupervised.adavae import (AdaVae, estimate_shared)
from disent.frameworks.unsupervised.vae import TrainingData, bce_loss_with_logits, kl_normal_loss
from disent.model import GaussianAutoEncoder
from disent.util import make_logger

log = make_logger()


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


@gin.configurable('framework.supervised.GuidedAdaVae')
class GuidedAdaVae(BaseFramework):

    MODE_ADAVAE = 'adavae'
    MODE_AVE_POS = 'ave_pos'
    MODE_AVE_TRIPLET = 'ave_triple'
    
    MODES = {MODE_AVE_TRIPLET}
    
    def __init__(
            self,
            beta=4,
            average_mode=AdaVae.AVE_MODE_GVAE,
            mode=MODE_AVE_TRIPLET,
            triplet_scale=0,
            triplet_alpha=0.3,
            triplet_after_sampling=False,
    ):
        # adavae instance
        assert average_mode == AdaVae.AVE_MODE_GVAE, f'currently only supports average_mode={repr(AdaVae.AVE_MODE_GVAE)}'
        self.adavae = AdaVae(beta=beta, average_mode=average_mode)
        # set mode
        assert mode in GuidedAdaVae.MODES, f'invalid {mode=}, must be one of {GuidedAdaVae.MODES}'
        self.mode = mode
        # use triplet loss
        self.triplet_scale = triplet_scale
        self.triplet_alpha = triplet_alpha
        self.triplet_after_sampling = triplet_after_sampling
        assert self.triplet_scale >= 0, f'triplet_scale={repr(self.triplet_scale)} must be non-negative'
        if self.triplet_scale > 0:
            assert mode == GuidedAdaVae.MODE_AVE_TRIPLET, f'triplet_scale={repr(self.triplet_scale)}, only supports triplet_scale > 0 for mode={repr(GuidedAdaVae.MODE_AVE_TRIPLET)}'
        # beta-vae stuffs
        self.beta = beta
    
    def training_step(self, model: GaussianAutoEncoder, batch):
        a_x, p_x, n_x = batch
        # ENCODE
        a_z_mean, a_z_logvar = model.encode_gaussian(a_x)
        p_z_mean, p_z_logvar = model.encode_gaussian(p_x)
        n_z_mean, n_z_logvar = model.encode_gaussian(n_x)
        # INTERCEPT
        (a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar), intercept_logs = self.intercept_z(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar)
        # REPARAMETERIZE
        a_z_sampled = model.reparameterize(a_z_mean, a_z_logvar)
        p_z_sampled = model.reparameterize(p_z_mean, p_z_logvar)
        n_z_sampled = model.reparameterize(n_z_mean, n_z_logvar)
        # RECONSTRUCT
        a_x_recon = model.decode(a_z_sampled)
        p_x_recon = model.decode(p_z_sampled)
        n_x_recon = model.decode(n_z_sampled)
        # COMPUTE LOSS
        loss_logs = self.compute_loss(
            TrainingData(a_x, a_x_recon, a_z_mean, a_z_logvar, a_z_sampled),
            TrainingData(p_x, p_x_recon, p_z_mean, p_z_logvar, p_z_sampled),
            TrainingData(n_x, n_x_recon, n_z_mean, n_z_logvar, n_z_sampled),
        )
        # RETURN INFO
        return {
            **intercept_logs,
            **loss_logs,
        }

    # @property
    # def required_observations(self):
    #     return 3

    def intercept_z(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        """
        *NB* arguments must satisfy: d(l, l2) < d(l, l3) [positive dist < negative dist]
        - This function assumes that the distance between labels l, l2, l3
          corresponding to z, z2, z3 satisfy the criteria d(l, l2) < d(l, l3)
          ie. l2 is the positive sample, l3 is the negative sample
        """
        # shared elements that need to be averaged, computed per pair in the batch.
        p_kl_deltas, p_kl_threshs, p_shared_mask = estimate_shared(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar)
        n_kl_deltas, n_kl_threshs, n_shared_mask = estimate_shared(a_z_mean, a_z_logvar, n_z_mean, n_z_logvar)

        # modify threshold based on criterion and recompute if necessary
        # CORE of this approach!
        old_p_ave_mask, old_n_ave_mask = p_shared_mask, n_shared_mask
        p_shared_mask, n_shared_mask = compute_constrained_masks(p_kl_deltas, old_p_ave_mask, n_kl_deltas, old_n_ave_mask)

        # if self.mode == 'adavae':
        #     new_args, _ = self.adavae.intercept_z(a_z_mean, a_z_logvar, p_z_mean, p_z_logvar)
        # elif self.mode == 'ave_pos':
        #     new_args = self.adavae.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), p_z_mean, p_z_logvar, p_shared_mask)
        if self.mode == 'ave_triple':
            (pAz_mean, pAz_logvar), (p_z_mean, p_z_logvar) = self.adavae.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), p_z_mean, p_z_logvar, p_shared_mask)
            (nAz_mean, nAz_logvar), (n_z_mean, n_z_logvar) = self.adavae.make_averaged(a_z_mean.clone(), a_z_logvar.clone(), n_z_mean, n_z_logvar, n_shared_mask)
            a_z_mean, a_z_logvar = self.adavae.compute_average(pAz_mean, pAz_logvar, nAz_mean, nAz_logvar)
            new_args = a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar
        else:
            raise KeyError

        return new_args, {
            'p_shared_before': old_p_ave_mask.sum() / len(a_z_mean),
            'p_shared_after': p_shared_mask.sum() / len(a_z_mean),
            'n_shared_before': old_n_ave_mask.sum() / len(a_z_mean),
            'n_shared_after': n_shared_mask.sum() / len(a_z_mean),
        }

    def compute_loss(self, a_data: TrainingData, p_data: TrainingData, n_data: TrainingData):
        # COMPUTE LOSS FOR TRIPLE:
        if self.mode == 'ave_triple':
            (a_x, a_x_recon, a_z_mean, a_z_logvar, a_z_sampled) = a_data
            (p_x, p_x_recon, p_z_mean, p_z_logvar, p_z_sampled) = p_data
            (n_x, n_x_recon, n_z_mean, n_z_logvar, n_z_sampled) = n_data
            
            # reconstruction error
            a_recon_loss = bce_loss_with_logits(a_x, a_x_recon)  # E[log p(x|z)]
            p_recon_loss = bce_loss_with_logits(p_x, p_x_recon)  # E[log p(x|z)]
            n_recon_loss = bce_loss_with_logits(n_x, n_x_recon)  # E[log p(x|z)]
            ave_recon_loss = (a_recon_loss + p_recon_loss + n_recon_loss) / 3

            # KL divergence
            a_kl_loss = kl_normal_loss(a_z_mean, a_z_logvar)  # D_kl(q(z|x) || p(z|x))
            p_kl_loss = kl_normal_loss(p_z_mean, p_z_logvar)  # D_kl(q(z|x) || p(z|x))
            n_kl_loss = kl_normal_loss(n_z_mean, n_z_logvar)  # D_kl(q(z|x) || p(z|x))
            ave_kl_loss = (a_kl_loss + p_kl_loss + n_kl_loss) / 3

            # compute combined loss
            loss = ave_recon_loss + self.beta * ave_kl_loss

            loss_dict = {
                'train_loss': loss,
                'reconstruction_loss': ave_recon_loss,
                'kl_loss': ave_kl_loss,
                'elbo': -(ave_recon_loss + ave_kl_loss),
            }

            if self.triplet_scale > 0:
                if self.triplet_after_sampling:
                    loss_triplet = triplet_loss(a_z_sampled, p_z_sampled, n_z_sampled, alpha=self.triplet_alpha)
                else:
                    loss_triplet = triplet_loss(a_z_mean, p_z_mean, n_z_mean, alpha=self.triplet_alpha)
                loss_dict.update({
                    'train_loss': loss + self.triplet_scale * loss_triplet,
                    'triplet_loss': loss_triplet
                })

        # COMPUTE LOSS FOR PAIR:
        # elif (self.mode == 'adavae') or (self.mode == 'ave_pos'):
        #     assert self.triplet_scale == 0, f'triplet_scale={repr(self.triplet_scale)}, triplet_scale > 0 is not supported for the current mode={self.mode}'
        #     loss_dict = self.adavae.compute_loss()
        else:
            raise KeyError

        return loss_dict


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def compute_constrained_masks(p_kl_deltas, p_shared_mask, n_kl_deltas, n_shared_mask):
    # number of changed factors
    p_shared_num = torch.sum(p_shared_mask, dim=1, keepdim=True)
    n_shared_num = torch.sum(n_shared_mask, dim=1, keepdim=True)
    
    # POSITIVE SHARED MASK
    # order from smallest to largest
    p_sort_indices = torch.argsort(p_kl_deltas, dim=1)
    # p_shared should be at least n_shared
    new_p_shared_num = torch.max(p_shared_num, n_shared_num)

    # NEGATIVE SHARED MASK
    # order from smallest to largest
    n_sort_indices = torch.argsort(n_kl_deltas, dim=1)
    # n_shared should be at most p_shared
    new_n_shared_num = torch.min(p_shared_num, n_shared_num)

    # COMPUTE NEW MASKS
    new_p_shared_mask = torch.zeros_like(p_shared_mask)
    new_n_shared_mask = torch.zeros_like(n_shared_mask)
    for i, (new_shared_p, new_shared_n) in enumerate(zip(new_p_shared_num, new_n_shared_num)):
        new_p_shared_mask[i, p_sort_indices[i, :new_shared_p]] = True
        new_n_shared_mask[i, n_sort_indices[i, :new_shared_n]] = True

    # return masks
    return new_p_shared_mask, new_n_shared_mask


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

