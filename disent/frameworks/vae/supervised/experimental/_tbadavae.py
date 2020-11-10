from dataclasses import dataclass

from disent.frameworks.vae.supervised import BoundedAdaVae
from disent.frameworks.vae.supervised._tvae import augment_loss_triplet


# ========================================================================= #
# tbadavae                                                                  #
# ========================================================================= #


class TripletBoundedAdaVae(BoundedAdaVae):

    @dataclass
    class Config(BoundedAdaVae.Config):
        # TODO: convert to triplet mixin
        triplet_margin: float = 0.1,
        triplet_scale: float = 1,
        triplet_p: int = 2,

    cfg: Config  # type hints

    def augment_loss(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        return augment_loss_triplet(a_z_mean, p_z_mean, n_z_mean, scale=self.cfg.triplet_scale, margin=self.cfg.triplet_margin, p=self.cfg.triplet_p)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
