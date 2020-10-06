import torch

from disent.frameworks.vae.supervised._tvae import TripletVae, augment_loss_triplet
from disent.frameworks.vae.weaklysupervised._adavae import AdaVae, compute_average_gvae


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AdaTripletVae(TripletVae):

    def __init__(self, *args, triplet_mode='ada', **kwargs):
        super().__init__(*args, **kwargs)
        # check modes
        assert triplet_mode in {'ada', 'p_ada', 'n_ada'}
        self.triplet_mode = triplet_mode

    def augment_loss(self, z_means, z_logvars, z_samples):
        a_z_mean, p_z_mean, n_z_mean = z_means
        a_z_logvar, p_z_logvar, n_z_logvar = z_logvars
        a_z_sampled, p_z_sampled, n_z_sampled = z_samples
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        trip_loss_old = augment_loss_triplet(a_z_mean, p_z_mean, n_z_mean, scale=self.triplet_scale, margin=self.triplet_margin)[0]
        trip_loss = triplet_loss(a_z_mean, p_z_mean, n_z_mean, margin=self.triplet_margin) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # estimate shared elements, then compute averaged vectors
        _, _, p_share_mask = AdaVae.estimate_shared(a_z_sampled, a_z_logvar, p_z_sampled, p_z_logvar)
        _, _, n_share_mask = AdaVae.estimate_shared(a_z_sampled, a_z_logvar, n_z_sampled, n_z_logvar)
        ap_ave_a_mean, ap_ave_a_logvar, ap_ave_p_mean, ap_ave_p_logvar = AdaVae.make_averaged(a_z_sampled, a_z_logvar, p_z_sampled, p_z_logvar, p_share_mask, compute_average_gvae)
        an_ave_a_mean, an_ave_a_logvar, an_ave_n_mean, an_ave_n_logvar = AdaVae.make_averaged(a_z_sampled, a_z_logvar, n_z_sampled, n_z_logvar, n_share_mask, compute_average_gvae)
        # get loss
        ada_triplet_loss = dist_triplet_loss(p_delta=ap_ave_a_mean-ap_ave_p_mean, n_delta=an_ave_a_mean-an_ave_n_mean, margin=self.triplet_margin) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # TEMP
        p_ada_trip_loss = triplet_loss(a=ap_ave_a_mean, p=ap_ave_p_mean, n=n_z_mean, margin=self.triplet_margin) * self.triplet_scale
        n_ada_trip_loss = triplet_loss(a=an_ave_a_mean, p=p_z_mean, n=an_ave_n_mean, margin=self.triplet_margin) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        loss = {'ada': ada_triplet_loss, 'p_ada': p_ada_trip_loss, 'n_ada': n_ada_trip_loss}[self.triplet_mode]
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        return loss, {
            'triplet_loss': trip_loss_old,
            'homebrew_triplet_loss': trip_loss,
            'ada_triplet_loss': ada_triplet_loss,
            'p_ada_trip_loss': p_ada_trip_loss,
            'n_ada_trip_loss': n_ada_trip_loss,
            # variance
            'a_var': torch.exp(a_z_logvar).mean(),
            'p_var': torch.exp(p_z_logvar).mean(),
            'n_var': torch.exp(n_z_logvar).mean(),
        }


def triplet_loss(a, p, n, margin=.1):
    return dist_triplet_loss(a-p, a-n, margin=margin)


def dist_triplet_loss(p_delta, n_delta, margin=1.):
    p_dist = torch.norm(p_delta, dim=-1)
    n_dist = torch.norm(n_delta, dim=-1)
    loss = torch.clamp_min(p_dist - n_dist + margin, 0)
    return loss.mean()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

