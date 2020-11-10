from dataclasses import dataclass

from disent.frameworks.vae.supervised import GuidedAdaVae
from disent.frameworks.vae.supervised._tvae import augment_loss_triplet


# ========================================================================= #
# tgadavae                                                                  #
# ========================================================================= #


class TripletGuidedAdaVae(GuidedAdaVae):

    @dataclass
    class Config(GuidedAdaVae.Config):
        # TODO: convert to triplet mixin
        triplet_margin = 0.1,
        triplet_scale = 1,
        triplet_p = 2,

    cfg: Config  # type hints

    def augment_loss(self, a_z_mean, a_z_logvar, p_z_mean, p_z_logvar, n_z_mean, n_z_logvar):
        return augment_loss_triplet(a_z_mean, p_z_mean, n_z_mean, scale=self.cfg.triplet_scale, margin=self.cfg.triplet_margin, p=self.cfg.triplet_p)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
