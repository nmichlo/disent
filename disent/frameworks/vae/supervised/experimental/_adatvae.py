import torch
import numpy as np

from disent.frameworks.vae.supervised._tvae import TripletVae, augment_loss_triplet
from disent.frameworks.vae.weaklysupervised._adavae import AdaVae, compute_average_gvae

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
            *args,
            average_mode=None,
            symmetric_kl=None,
            triplet_mode='ada_p_orig_lerp',
            triplet_l=1,
            # lerp
            lerp_steps=10000,  # 12*3600 | 50400 = 14*3600
            steps_offset=0,
            lerp_goal=1.0,
            **kwargs
    ):
        # check symmetric_kl arg
        if symmetric_kl is not None:
            log.warning(f'{symmetric_kl=} argument is unused, always symmetric for {AdaTripletVae.__name__}!')
        if average_mode is not None:
            log.warning(f'{average_mode=} argument is unused, always gvae averaging for {AdaTripletVae.__name__}!')

        # initialise
        super().__init__(*args, **kwargs)

        # triplet loss mode
        self.triplet_mode = triplet_mode
        self.triplet_l = triplet_l

        # triplet annealing
        self.lerp_steps = lerp_steps
        self.steps = 0
        self.steps_offset = steps_offset
        self.lerp_goal = lerp_goal

    def augment_loss(self, z_means, z_logvars, z_samples):
        a_z_mean, p_z_mean, n_z_mean = z_means

        # normal triplet
        trip_loss = triplet_loss(a_z_mean, p_z_mean, n_z_mean, margin=self.triplet_margin, l=self.triplet_l) * self.triplet_scale

        # Adaptive Component
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # TODO: good reason why `ap_p_ave - ap_a_ave` this is bad?
        _, _, an_a_ave, an_n_ave = AdaTripletVae.compute_ave(a_z_mean, p_z_mean, n_z_mean)
        ada_p_orig = dist_triplet_loss(p_delta=a_z_mean - p_z_mean, n_delta=an_a_ave - an_n_ave, margin=self.triplet_margin, l=self.triplet_l) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        # Triplet Lerp
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.steps += 1
        lerp = np.clip((self.steps - self.steps_offset) / self.lerp_steps, 0, self.lerp_goal)
        # TODO: good reason why `ap_p_ave - ap_a_ave` this is bad?
        _, _, an_a_ave, an_n_ave = AdaTripletVae.compute_ave(a_z_mean, p_z_mean, n_z_mean, lerp=lerp)
        ada_p_orig_lerp = dist_triplet_loss(p_delta=a_z_mean - p_z_mean, n_delta=an_a_ave - an_n_ave, margin=self.triplet_margin, l=self.triplet_l) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

        losses = {
            # normal
            'triplet': trip_loss,
            # best
            'ada_p_orig': ada_p_orig,
            'ada_p_orig_lerp': ada_p_orig_lerp,  # BEST!
            # OLD
            'ada_p_orig_and_trip': (self.lerp_goal * ada_p_orig) + ((1-self.lerp_goal) * trip_loss),
            'ada_p_orig_and_trip_lerp': (self.lerp_goal * ada_p_orig_lerp) + ((1-self.lerp_goal) * trip_loss),
            # lerp
            'trip_lerp_ada_p_orig': (lerp * ada_p_orig) + ((1-lerp) * trip_loss),
            'trip_lerp_ada_p_orig_lerp': (lerp * ada_p_orig_lerp) + ((1-lerp) * trip_loss),
        }

        return losses[self.triplet_mode], {
            **losses,
            'lerp': lerp,
            'lerp_goal': self.lerp_goal,
        }

    @staticmethod
    def compute_ave(a_z_mean, p_z_mean, n_z_mean, lerp=None):
        # ADAPTIVE COMPONENT
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        delta_p = torch.abs(a_z_mean - p_z_mean)
        delta_n = torch.abs(a_z_mean - n_z_mean)
        # get thresholds
        p_thresh = AdaVae.estimate_threshold(delta_p, keepdim=True)
        n_thresh = AdaVae.estimate_threshold(delta_n, keepdim=True)
        # interpolate threshold
        if lerp is not None:
            p_thresh = (lerp * p_thresh) + ((1-lerp) * torch.min(delta_p, dim=-1, keepdim=True).values)
            n_thresh = (lerp * n_thresh) + ((1-lerp) * torch.min(delta_n, dim=-1, keepdim=True).values)
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
        # compute averaged
        ap_ave = (0.5 * a_z_mean) + (0.5 * p_z_mean)
        an_ave = (0.5 * a_z_mean) + (0.5 * n_z_mean)
        ap_a_ave, ap_p_ave = torch.where(p_shared, ap_ave, a_z_mean), torch.where(p_shared, ap_ave, p_z_mean)
        an_a_ave, an_n_ave = torch.where(n_shared, an_ave, a_z_mean), torch.where(n_shared, an_ave, n_z_mean)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        return ap_a_ave, ap_p_ave, an_a_ave, an_n_ave


def triplet_loss(a, p, n, margin=.1, l=1, top_k=None):
    return dist_triplet_loss(a - p, a - n, margin=margin, l=l, top_k=top_k)


def dist_triplet_loss(p_delta, n_delta, margin=1., l=1, top_k=None):
    p_dist = torch.norm(p_delta, p=l, dim=-1)
    n_dist = torch.norm(n_delta, p=l, dim=-1)
    loss = torch.clamp_min(p_dist - n_dist + margin, 0)
    if top_k is not None:
        loss = torch.topk(loss, k=top_k).values
    return loss.mean()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #



# -m dataset=xysquares framework=adatvae