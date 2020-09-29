import cv2
import torch
import numpy as np
import torchvision
from pl_bolts.models.self_supervised import SimCLR
from disent.frameworks.framework import BaseFramework


# ========================================================================= #
# simclr                                                                   #
# ========================================================================= #


class AdaSimCLR(SimCLR, BaseFramework):
    # https://github.com/Spijkervet/SimCLR
    # https://github.com/leftthomas/SimCLR
    # https://github.com/sthalles/SimCLR
    # https://github.com/sthalles/PyTorch-BYOL

    def __init__(self, make_optimizer_fn, batch_augment):
        super().__init__(make_optimizer_fn, batch_augment=batch_augment)

    def compute_training_loss(self, batch, batch_idx) -> dict:
        pass

    def shared_step(self, batch, batch_idx):
        (img1, img2), y = batch

        # ENCODE
        # encode -> representations
        # (b, 3, 32, 32) -> (b, 2048, 2, 2)
        h1 = self.encoder(img1)
        h2 = self.encoder(img2)

        # the bolts resnets return a list of feature maps
        if isinstance(h1, list):
            h1 = h1[-1]
            h2 = h2[-1]

        # PROJECT
        # img -> E -> h -> || -> z
        # (b, 2048, 2, 2) -> (b, 128)
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2, self.hparams.loss_temperature)

        return loss


# ========================================================================= #
# SimCLR TRANSFORMS                                                         #
# ========================================================================= #


def make_simclr_transforms(size, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    return torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            torchvision.transforms.RandomApply([GaussianBlurTransform(int(size[1] * 0.1))], p=0.5),  # kernel size should be set to 10% of the image size
            torchvision.transforms.ToTensor(),
    ])


class GaussianBlurTransform(object):
    """
    Gaussian blur as described in the SimCLR paper (without the random application chance).
    - had random chance to choose blur radius
    - https://github.com/sthalles/SimCLR
    """

    def __init__(self, kernel_size, sigma_min=0.1, sigma_max=2.0):
        self.kernel_size = kernel_size
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max

    def __call__(self, sample):
        # TODO: fix casting. Is this actually needed?
        # sample = np.array(sample)
        assert sample.dtype == np.float32  # or sample.dtype == np.float16
        sigma = (self._sigma_max - self._sigma_min) * np.random.random_sample() + self._sigma_min
        return cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
