from dataclasses import dataclass
from disent.frameworks.vae.supervised import BoundedAdaVae, TripletVae
from disent.frameworks.helper.triplet_loss import TripletLossConfig


# ========================================================================= #
# tbadavae                                                                  #
# ========================================================================= #


class TripletBoundedAdaVae(BoundedAdaVae):

    @dataclass
    class cfg(BoundedAdaVae.cfg, TripletLossConfig):
        pass

    def augment_loss(self, z_means):
        return TripletVae.augment_loss_triplet(z_means, self.cfg)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
