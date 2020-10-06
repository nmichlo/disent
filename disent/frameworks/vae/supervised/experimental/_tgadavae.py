from disent.frameworks.vae.supervised import GuidedAdaVae
from disent.frameworks.vae.supervised._tvae import augment_loss_triplet


# ========================================================================= #
# tgadavae                                                                  #
# ========================================================================= #


class TripletGuidedAdaVae(GuidedAdaVae):
    
    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            batch_augment=None,
            beta=4,
            average_mode='gvae',
            anchor_ave_mode='average',
            triplet_margin=0.1,
            triplet_scale=1,
    ):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, beta=beta, average_mode=average_mode, anchor_ave_mode=anchor_ave_mode)
        self.triplet_margin = triplet_margin
        self.triplet_scale = triplet_scale

    def augment_loss(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        return augment_loss_triplet(a_z_mean, p_z_mean, n_z_mean, scale=self.triplet_scale, margin=self.triplet_margin)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
