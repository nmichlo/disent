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
                kornia.augmentation.RandomCrop(size=size, padding=5),
                kornia.augmentation.RandomPerspective(distortion_scale=0.05, p=1.0),
                kornia.augmentation.RandomRotation(degrees=4),
            ])

        # generate augmented items
        with torch.no_grad():
            p_x_targ = a_x_targ
            p_x = self._aug(a_x)
            a_x = self._aug(a_x)
            n_x = self._aug(n_x)

        batch['x'], batch['x_targ'] = (a_x, p_x, n_x), (a_x_targ, p_x_targ, n_x_targ)
        # compute!
        return super().compute_training_loss(batch, batch_idx)

    # def augment_loss(self, z_means, z_logvars, z_samples):
    #     a_z_mean, p_z_mean, n_z_mean = z_means
    #     # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    #
    #     # normal triplet
    #     trip_loss = triplet_loss(a_z_mean, p_z_mean, n_z_mean, margin=self.triplet_margin) * self.triplet_scale
    #
    #     # Adaptive Component
    #     # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    #     # TODO: good reason why `ap_p_ave - ap_a_ave` this is bad?
    #     _, _, an_a_ave, an_n_ave = AdaTripletVae.compute_ave(a_z_mean, p_z_mean, n_z_mean)
    #     ada_p_orig = dist_triplet_loss(p_delta=a_z_mean-p_z_mean, n_delta=an_a_ave-an_n_ave, margin=self.triplet_margin) * self.triplet_scale
    #     # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    #
    #
    #     # Triplet Lerp
    #     # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    #     self.steps += 1
    #     lerp = np.clip((self.steps - self.steps_offset) / self.lerp_steps, 0, self.lerp_goal)
    #     # TODO: good reason why `ap_p_ave - ap_a_ave` this is bad?
    #     _, _, an_a_ave, an_n_ave = AdaTripletVae.compute_ave(a_z_mean, p_z_mean, n_z_mean, lerp=lerp)
    #     ada_p_orig_lerp = dist_triplet_loss(p_delta=a_z_mean-p_z_mean, n_delta=an_a_ave-an_n_ave, margin=self.triplet_margin) * self.triplet_scale
    #     # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
    #
    #     losses = {
    #         'ada_p_orig': ada_p_orig,
    #         'ada_p_orig_lerp': ada_p_orig_lerp,  # BEST!
    #         # OLD
    #         'ada_p_orig_and_trip': (self.lerp_goal * ada_p_orig) + ((1-self.lerp_goal) * trip_loss),
    #         'ada_p_orig_and_trip_lerp': (self.lerp_goal * ada_p_orig_lerp) + ((1-self.lerp_goal) * trip_loss),
    #         # lerp
    #         'trip_lerp_ada_p_orig': (lerp * ada_p_orig) + ((1-lerp) * trip_loss),
    #         'trip_lerp_ada_p_orig_lerp': (lerp * ada_p_orig_lerp) + ((1-lerp) * trip_loss),
    #     }
    #
    #     return losses[self.triplet_mode], {
    #         'triplet_loss': trip_loss,
    #         **losses,
    #         'lerp': lerp,
    #         'lerp_goal': self.lerp_goal,
    #     }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

