import logging
import kornia
import torch
import torchvision

from disent.frameworks.vae.supervised._tvae import TripletVae


log = logging.getLogger(__name__)


# ========================================================================= #
# Guided Ada Vae                                                            #
# ========================================================================= #


class AugPosTripletVae(TripletVae):

    def __init__(
            self,
            make_optimizer_fn,
            make_model_fn,
            batch_augment=None,
            beta=4,
            # tvae: triplet stuffs
            triplet_margin=10,
            triplet_scale=100,
            triplet_p=2,
            # tvae: no loss from decoder -> encoder
            detach=False,
            detach_decoder=True,
            detach_no_kl=False,
            detach_logvar=-2,
    ):
        super().__init__(
            make_optimizer_fn,
            make_model_fn,
            batch_augment=batch_augment,
            beta=beta,
            triplet_margin=triplet_margin,
            triplet_scale=triplet_scale,
            triplet_p=triplet_p,
            detach=detach,
            detach_decoder=detach_decoder,
            detach_no_kl=detach_no_kl,
            detach_logvar=detach_logvar,
        )
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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

