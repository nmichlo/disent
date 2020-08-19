import torch
import torchvision

from disent.dataset.transforms import GaussianBlurTransform
from disent.frameworks.framework import BaseFramework


# ========================================================================= #
# simclr                                                                   #
# ========================================================================= #


class SimCLR(BaseFramework):
    # https://github.com/Spijkervet/SimCLR
    # https://github.com/leftthomas/SimCLR
    # https://github.com/sthalles/SimCLR
    # https://github.com/sthalles/PyTorch-BYOL
    
    def __init__(self, make_optimizer_fn):
        super().__init__(make_optimizer_fn)

    def forward(self, batch) -> torch.Tensor:
        pass

    def compute_loss(self, batch, batch_idx) -> dict:
        pass


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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
