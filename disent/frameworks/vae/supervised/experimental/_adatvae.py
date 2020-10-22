import torch
import numpy as np

from disent.frameworks.vae.supervised._tvae import TripletVae, triplet_loss, dist_triplet_loss
from disent.frameworks.vae.weaklysupervised._adavae import AdaVae

import logging

log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AdaTripletVae(TripletVae):

    # TODO: increase margin over time, maybe log
    #       approach current max

    # TODO: given reconstruction loss, can we use it as a signal if things are going badly.
    #      validation signal? Intelligent way we can use this?

    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            batch_augment=None,
            beta=4,
            # tvae: triplet stuffs
            triplet_margin=10,
            triplet_scale=100,
            triplet_p=2,
            # tvae: no loss from decoder -> encoder
            detach=False,
            detach_decoder=True,
            detach_no_kl=False,
            detach_logvar=-2,
            # adatvae: what version of triplet to use
            triplet_mode='ada_p_orig_lerp',
            # adatvae: annealing
            lerp_steps=10000,  # 12*3600 | 50400 = 14*3600
            steps_offset=0,
            lerp_goal=1.0,
    ):

        # initialise
        super().__init__(
            make_optimizer_fn,
            make_model_fn,
            batch_augment=batch_augment,
            beta=beta,
            triplet_margin=triplet_margin,
            triplet_scale=triplet_scale,
            triplet_p=triplet_p,
            detach=detach,
            detach_decoder=detach_decoder,
            detach_no_kl=detach_no_kl,
            detach_logvar=detach_logvar,
        )

        # triplet loss mode
        self.triplet_mode = triplet_mode
        self.triplet_p = triplet_p

        # triplet annealing
        self.lerp_steps = lerp_steps
        self.steps = 0
        self.steps_offset = steps_offset
        self.lerp_goal = lerp_goal

    def augment_loss(self, z_means, z_logvars, z_samples):
        a_z_mean, p_z_mean, n_z_mean = z_means

        # normal triplet
        trip_loss = triplet_loss(a_z_mean, p_z_mean, n_z_mean, margin=self.triplet_margin, p=self.triplet_p) * self.triplet_scale

        # Adaptive Component
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # TODO: good reason why `ap_p_ave - ap_a_ave` this is bad?
        _, _, an_a_ave, an_n_ave = AdaTripletVae.compute_ave(a_z_mean, p_z_mean, n_z_mean)
        ada_p_orig = dist_triplet_loss(pos_delta=p_z_mean-a_z_mean, neg_delta=an_n_ave-an_a_ave, margin=self.triplet_margin, p=self.triplet_p) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # Update Anneal Values
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.steps += 1
        lerp = (self.steps - self.steps_offset) / self.lerp_steps
        lerp = np.clip(lerp, 0, self.lerp_goal)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # Triplet Lerp
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # TODO: good reason why `ap_p_ave - ap_a_ave` this is bad?
        _, _, an_a_ave, an_n_ave = AdaTripletVae.compute_ave(a_z_mean, p_z_mean, n_z_mean, lerp=lerp)
        ada_p_orig_lerp = dist_triplet_loss(pos_delta=p_z_mean-a_z_mean, neg_delta=an_n_ave-an_a_ave, margin=self.triplet_margin, p=self.triplet_p) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # MSE Ada Triplet
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        p_shared_mask, n_shared_mask = AdaTripletVae.compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=None)
        p_shared_mask_lerp, n_shared_mask_lerp = AdaTripletVae.compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=lerp)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # TODO: does p here affect things?
        shared_loss = AdaTripletVae.compute_shared_loss(a_z_mean, p_z_mean, n_z_mean, lerp=None, p=2) * self.triplet_scale
        shared_loss_lerp = AdaTripletVae.compute_shared_loss(a_z_mean, p_z_mean, n_z_mean, lerp=lerp, p=2) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # TODO: try triplet but instead of adding MSE, multiply the shared deltas
        #  elements so they are moved closer together. ie. 2x for a->p, and 0.5x for a->n
        # Ada Mul Triplet
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        mul = torch.where(p_shared_mask, torch.full_like(a_z_mean, 2), torch.full_like(a_z_mean, 1))
        ada_mul_triplet = dist_triplet_loss(pos_delta=(p_z_mean-a_z_mean) * mul, neg_delta=(n_z_mean-a_z_mean) / mul, margin=self.triplet_margin, p=self.triplet_p) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        mul = torch.where(p_shared_mask, torch.full_like(a_z_mean, 1+lerp), torch.full_like(a_z_mean, 1))
        ada_mul_lerp_triplet = dist_triplet_loss(pos_delta=(p_z_mean-a_z_mean) * mul, neg_delta=(n_z_mean-a_z_mean) / mul, margin=self.triplet_margin, p=self.triplet_p) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        mul = torch.where(p_shared_mask_lerp, torch.full_like(a_z_mean, 2), torch.full_like(a_z_mean, 1))
        ada_lerp_mul_triplet = dist_triplet_loss(pos_delta=(p_z_mean-a_z_mean) * mul, neg_delta=(n_z_mean-a_z_mean) / mul, margin=self.triplet_margin, p=self.triplet_p) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        mul = torch.where(p_shared_mask_lerp, torch.full_like(a_z_mean, 1+lerp), torch.full_like(a_z_mean, 1))
        ada_lerp_mul_lerp_triplet = dist_triplet_loss(pos_delta=(p_z_mean-a_z_mean) * mul, neg_delta=(n_z_mean-a_z_mean) / mul, margin=self.triplet_margin, p=self.triplet_p) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        losses = {
            # normal
            'triplet': trip_loss,
            # Ada Mul Triplet
            'ada_mul_triplet': ada_mul_triplet,
            'ada_mul_lerp_triplet': ada_mul_lerp_triplet,
            'ada_lerp_mul_triplet': ada_lerp_mul_triplet,
            'ada_lerp_mul_lerp_triplet': ada_lerp_mul_lerp_triplet,
            # MSE adaptive triplet
            'trip_and_mse_ada':          blend(trip_loss, shared_loss,      alpha=0.5),  # too strong
            'trip_and_mse_ada_lerp':     blend(trip_loss, shared_loss_lerp, alpha=0.5),
            'lerp_trip_to_mse_ada':      blend(trip_loss, shared_loss,      alpha=lerp),
            'lerp_trip_to_mse_ada_lerp': blend(trip_loss, shared_loss_lerp, alpha=lerp),
            # best
            'ada_p_orig':      ada_p_orig,
            'ada_p_orig_lerp': ada_p_orig_lerp,  # BEST!
            # OLD
            'trip_and_ada_p_orig':          blend(trip_loss, ada_p_orig,      alpha=0.5),
            'trip_and_ada_p_orig_lerp':     blend(trip_loss, ada_p_orig_lerp, alpha=0.5),
            'lerp_trip_to_ada_p_orig':      blend(trip_loss, ada_p_orig,      alpha=lerp),
            'lerp_trip_to_ada_p_orig_lerp': blend(trip_loss, ada_p_orig_lerp, alpha=lerp),
        }

        return losses[self.triplet_mode], {
            **losses,
            'triplet_chosen': losses[self.triplet_mode],
            # lerp
            'lerp': lerp,
            'lerp_goal': self.lerp_goal,
            # shared
            'p_shared': p_shared_mask.sum(dim=1).float().mean(),
            'n_shared': n_shared_mask.sum(dim=1).float().mean(),
            'p_shared_lerp': p_shared_mask_lerp.sum(dim=1).float().mean(),
            'n_shared_lerp': n_shared_mask_lerp.sum(dim=1).float().mean(),
        }

    @staticmethod
    def compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=None):
        # ADAPTIVE COMPONENT
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        delta_p = torch.abs(a_z_mean - p_z_mean)
        delta_n = torch.abs(a_z_mean - n_z_mean)
        # get thresholds
        p_thresh = AdaVae.estimate_threshold(delta_p, keepdim=True)
        n_thresh = AdaVae.estimate_threshold(delta_n, keepdim=True)
        # interpolate threshold
        if lerp is not None:
            p_thresh = blend(torch.min(delta_p, dim=-1, keepdim=True).values, p_thresh, alpha=lerp)
            n_thresh = blend(torch.min(delta_n, dim=-1, keepdim=True).values, n_thresh, alpha=lerp)
            # -------------- #
            # # RANDOM LERP:
            # # This should average out to the value given above
            # p_min, p_max = torch.min(delta_p, dim=-1, keepdim=True).values, torch.max(delta_p, dim=-1, keepdim=True).values
            # n_min, n_max = torch.min(delta_n, dim=-1, keepdim=True).values, torch.max(delta_n, dim=-1, keepdim=True).values
            # p_thresh = p_min + torch.rand_like(p_thresh) * (p_max - p_min) * lerp
            # n_thresh = p_min + torch.rand_like(n_thresh) * (n_max - n_min) * lerp
            # -------------- #
        # estimate shared elements, then compute averaged vectors
        p_shared = (delta_p < p_thresh).detach()
        n_shared = (delta_n < n_thresh).detach()
        # done!
        return p_shared, n_shared

    @staticmethod
    def compute_ave(a_z_mean, p_z_mean, n_z_mean, lerp=None):
        # estimate shared elements, then compute averaged vectors
        p_shared, n_shared = AdaTripletVae.compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=lerp)
        # compute averaged
        ap_ave = (0.5 * a_z_mean) + (0.5 * p_z_mean)
        an_ave = (0.5 * a_z_mean) + (0.5 * n_z_mean)
        ap_a_ave, ap_p_ave = torch.where(p_shared, ap_ave, a_z_mean), torch.where(p_shared, ap_ave, p_z_mean)
        an_a_ave, an_n_ave = torch.where(n_shared, an_ave, a_z_mean), torch.where(n_shared, an_ave, n_z_mean)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        return ap_a_ave, ap_p_ave, an_a_ave, an_n_ave

    @staticmethod
    def compute_shared_loss(a_z_mean, p_z_mean, n_z_mean, lerp=None, p=2):
        p_shared_mask, n_shared_mask = AdaTripletVae.compute_shared_masks(a_z_mean, p_z_mean, n_z_mean, lerp=lerp)
        p_shared_loss = torch.norm(torch.where(p_shared_mask, a_z_mean-p_z_mean, torch.zeros_like(a_z_mean)), p=p, dim=-1).mean()
        n_shared_loss = torch.norm(torch.where(n_shared_mask, a_z_mean-n_z_mean, torch.zeros_like(a_z_mean)), p=p, dim=-1).mean()
        shared_loss = 0.5 * p_shared_loss + 0.5 * n_shared_loss
        return shared_loss


def blend(a, b, alpha):
    """
    if alpha == 0 then a is returned
    if alpha == 1 then b is returned
    """
    alpha = np.clip(alpha, 0, 1)
    return ((1-alpha) * a) + (alpha * b)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
