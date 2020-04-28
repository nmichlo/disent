import torch.nn as nn
import numpy as np


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #
from torch import Tensor


class View(nn.Module):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, *size):
        super().__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(*self.size)

class Unsqueeze3D(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return x

class Flatten3D(nn.Module):
    """From: https://github.com/amir-abdi/disentanglement-pytorch"""
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class BaseModule(nn.Module):
    def __init__(self, x_shape=(3, 64, 64), z_size=6):
        super().__init__()
        self._x_shape = x_shape
        self._x_size = int(np.prod(x_shape))
        self._z_size = z_size

    @property
    def x_shape(self):
        return self._x_shape
    @property
    def x_size(self):
        return self._x_size
    @property
    def z_size(self):
        return self._z_size

    def assert_x_valid(self, x):
        assert x.ndim == 4
        assert x.shape[1:] == self.x_shape
    def assert_z_valid(self, z):
        assert z.ndim == 2
        assert z.shape[1] == self.z_size
    def assert_lengths(self, x, z):
        assert len(x) == len(z)


class BaseEncoderModule(BaseModule):
    def forward(self, x) -> Tensor:
        self.assert_x_valid(x)
        # encode
        z = self.encode(x)
        # checks
        self.assert_z_valid(z)
        self.assert_lengths(x, z)
        # return
        return z

    def encode(self, x) -> Tensor:
        raise NotImplementedError


class BaseGaussianEncoderModule(BaseModule):
    def forward(self, x) -> (Tensor, Tensor):
        self.assert_x_valid(x)
        # encode | p(z|x)
        z_mean, z_logvar = self.encode_gaussian(x)
        # checks
        self.assert_z_valid(z_mean)
        self.assert_z_valid(z_logvar)
        self.assert_lengths(x, z_logvar)
        self.assert_lengths(x, z_mean)
        # return
        return z_mean, z_logvar

    def encode_gaussian(self, x) -> (Tensor, Tensor):
        raise NotImplementedError


class BaseDecoderModule(BaseModule):
    def forward(self, z):
        self.assert_z_valid(z)
        # decode | p(x|z)
        x_recon = self.decode(z)
        # checks
        self.assert_x_valid(x_recon)
        self.assert_lengths(x_recon, z)
        # return
        return x_recon

    def decode(self, z) -> Tensor:
        raise NotImplementedError


# class BaseAutoEncoderModule(BaseEncoderModule, BaseDecoderModule):
#     def forward(self, x):
#         self.assert_x_valid(x)
#         # encode
#         z = self.encode(x)
#         self.assert_z_valid(z)
#         assert len(x) == len(z)
#         # decode
#         x_recon = self.decode(z)
#         self.assert_x_valid(x_recon)
#         assert len(z) == len(x_recon)
#         # return
#         return x_recon


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
