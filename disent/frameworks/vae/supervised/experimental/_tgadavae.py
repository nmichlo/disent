from dataclasses import dataclass
from disent.frameworks.vae.supervised import GuidedAdaVae
from disent.frameworks.vae.supervised._tvae import TripletVae
from disent.frameworks.helper.triplet_loss import TripletLossConfig


# ========================================================================= #
# tgadavae                                                                  #
# ========================================================================= #


class TripletGuidedAdaVae(GuidedAdaVae):

    @dataclass
    class cfg(GuidedAdaVae.cfg, TripletLossConfig):
        pass

    def augment_loss(self, z_means):
        return TripletVae.augment_loss_triplet(z_means, self.cfg)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
