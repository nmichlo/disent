import torch.nn as nn
from torch import Tensor
from disent.model.base import BaseGaussianEncoderModule, BaseDecoderModule, Flatten3D, Unsqueeze3D, View


# ========================================================================= #
# simple 28x28 fully connected models                                       #
# ========================================================================= #


class EncoderSimpleFC(BaseGaussianEncoderModule):

    def __init__(self, x_shape=(3, 64, 64), h_size1=128, h_size2=128, z_size=6):
        super().__init__(x_shape=x_shape, z_size=z_size)
        self.model = nn.Sequential(
            Flatten3D(),
            nn.Linear(self.x_size, h_size1),
            nn.ReLU(True),
            nn.Linear(h_size1, h_size2),
            nn.ReLU(True),
        )
        self.enc3mean = nn.Linear(h_size2, self.z_size)
        self.enc3logvar = nn.Linear(h_size2, self.z_size)

    def encode_gaussian(self, x) -> (Tensor, Tensor):
        pre_z = self.model(x)
        return self.enc3mean(pre_z), self.enc3logvar(pre_z)


class DecoderSimpleFC(BaseDecoderModule):

    def __init__(self, x_shape=(3, 64, 64), h_size1=128, h_size2=128, z_size=6):
        super().__init__(x_shape=x_shape, z_size=z_size)
        self.model = nn.Sequential(
            nn.Linear(self.z_size, h_size2),
            nn.ReLU(True),
            nn.Linear(h_size2, h_size1),
            nn.ReLU(True),
            nn.Linear(h_size1, self.x_size),
            View(-1, *self.x_shape),
        )

    def decode(self, z):
        return self.model(z)


# ========================================================================= #
# simple 64x64 convolutional models                                         #
# ========================================================================= #


class EncoderSimpleConv64(BaseGaussianEncoderModule):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""

    def __init__(self, x_shape=(3, 64, 64), z_size=6):
        # checks
        assert len(x_shape) == 3
        img_channels, image_h, image_w = x_shape
        assert (image_h == 64) and (image_w == 64), 'This model only works with image size 64x64.'
        # initialise
        super().__init__(x_shape=x_shape, z_size=z_size)
        #
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            Flatten3D(),
        )
        self.enc3mean = nn.Linear(256, self.z_size)
        self.enc3logvar = nn.Linear(256, self.z_size)

    def forward(self, x):
        x = self.model(x)
        return self.enc3mean(x), self.enc3logvar(x)


class DecoderSimpleConv64(BaseDecoderModule):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""

    def __init__(self, x_shape=(3, 64, 64), z_size=6):
        # checks
        assert len(x_shape) == 3
        img_channels, image_h, image_w = x_shape
        assert (image_h == 64) and (image_w == 64), 'This model only works with image size 64x64.'
        # initialise
        super().__init__(x_shape=x_shape, z_size=z_size)

        self.model = nn.Sequential(
            Unsqueeze3D(),
            nn.Conv2d(img_channels, 256, 1, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 128, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 3, 1)
        )
        # output shape = bs x 3 x 64 x 64

    def forward(self, x):
        return self.model(x)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
