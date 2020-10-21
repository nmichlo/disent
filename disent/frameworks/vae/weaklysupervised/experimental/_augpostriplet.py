import logging
import kornia
import torch
import torchvision

from disent.frameworks.vae.supervised._tvae import TripletVae
from disent.frameworks.vae.supervised.experimental._adatvae import triplet_loss_l2, triplet_loss_l1

log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AugPosTripletVae(TripletVae):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._aug = None

    def compute_training_loss(self, batch, batch_idx):
        (a_x, n_x), (a_x_targ, n_x_targ) = batch['x'], batch['x_targ']

        # make augmenter as it requires the image sizes
        if self._aug is None:
            size = a_x.shape[2:4]
            self._aug = torchvision.transforms.RandomOrder([
                kornia.augmentation.ColorJitter(brightness=0.25, contrast=0.25, saturation=0, hue=0.15),
                kornia.augmentation.RandomCrop(size=size, padding=8),
                # kornia.augmentation.RandomPerspective(distortion_scale=0.05, p=1.0),
                # kornia.augmentation.RandomRotation(degrees=4),
            ])

        # generate augmented items
        with torch.no_grad():
            p_x_targ = a_x_targ
            p_x = self._aug(a_x)
            # a_x = self._aug(a_x)
            # n_x = self._aug(n_x)

        batch['x'], batch['x_targ'] = (a_x, p_x, n_x), (a_x_targ, p_x_targ, n_x_targ)
        # compute!
        return super().compute_training_loss(batch, batch_idx)

    def augment_loss(self, z_means, z_logvars, z_samples):
        a_z_mean, p_z_mean, n_z_mean = z_means
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        trip_loss_l2 = triplet_loss_l2(a_z_mean, p_z_mean, n_z_mean, margin=self.triplet_margin) * self.triplet_scale
        trip_loss_l1 = triplet_loss_l1(a_z_mean, p_z_mean, n_z_mean, margin=self.triplet_margin) * self.triplet_scale
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        loss = trip_loss_l1
        return loss, {
            'triplet_loss_l2': trip_loss_l2,
            'triplet_loss_l1': trip_loss_l1,
        }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

