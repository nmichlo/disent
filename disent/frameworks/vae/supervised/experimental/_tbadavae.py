from disent.frameworks.vae.supervised import BoundedAdaVae
from disent.frameworks.vae.supervised._tvae import augment_loss_triplet


# ========================================================================= #
# tbadavae                                                                  #
# ========================================================================= #


class TripletBoundedAdaVae(BoundedAdaVae):

    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            batch_augment=None,
            beta=4,
            average_mode='gvae',
            symmetric_kl=True,
            triplet_margin=0.1,
            triplet_scale=1,
    ):
        super().__init__(make_optimizer_fn, make_model_fn, batch_augment=batch_augment, beta=beta, average_mode=average_mode, symmetric_kl=symmetric_kl)
        self.triplet_margin = triplet_margin
        self.triplet_scale = triplet_scale

    def augment_loss(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        return augment_loss_triplet(a_z_mean, p_z_mean, n_z_mean, scale=self.triplet_scale, margin=self.triplet_margin)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
