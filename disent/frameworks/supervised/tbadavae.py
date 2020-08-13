from disent.frameworks.supervised.badavae import BoundedAdaVae
from disent.frameworks.supervised.tgadavae import triplet_loss


# ========================================================================= #
# tbadavae                                                                  #
# ========================================================================= #


class TripletBoundedAdaVae(BoundedAdaVae):

    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            beta=4,
            average_mode='gvae',
            triplet_alpha=0.1,
            triplet_scale=1,
    ):
        super().__init__(make_optimizer_fn, make_model_fn, beta=beta, average_mode=average_mode)
        self.triplet_alpha = triplet_alpha
        self.triplet_scale = triplet_scale

    def augment_loss(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        loss_triplet = triplet_loss(a_z_mean, p_z_mean, n_z_mean, alpha=self.triplet_alpha)
        augmented_loss = self.triplet_scale * loss_triplet
        return augmented_loss, {'triplet_loss': loss_triplet}


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
