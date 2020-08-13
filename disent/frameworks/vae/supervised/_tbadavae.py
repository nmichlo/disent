from disent.frameworks.vae.supervised._tgadavae import triplet_loss
from disent.frameworks.vae.supervised import BoundedAdaVae
import torch.nn.functional as F

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
            mse_shared_loss_weight=0,
            triplet_margin=0.1,
            triplet_scale=1,
    ):
        super().__init__(make_optimizer_fn, make_model_fn, beta=beta, average_mode=average_mode, mse_shared_loss_weight=mse_shared_loss_weight)
        self.triplet_margin = triplet_margin
        self.triplet_scale = triplet_scale

    def augment_loss(self, a_z_mean_old, a_z_logvar_old, p_z_mean_old, p_z_logvar_old, n_z_mean_old, n_z_logvar_old):
        loss_triplet = triplet_loss(a_z_mean_old, p_z_mean_old, n_z_mean_old, margin=self.triplet_margin)
        augmented_loss = self.triplet_scale * loss_triplet
        return augmented_loss, {
            'triplet_loss': loss_triplet,
            'triplet_loss_torch': F.triplet_margin_loss(a_z_mean_old, p_z_mean_old, n_z_mean_old, margin=self.triplet_margin)
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
