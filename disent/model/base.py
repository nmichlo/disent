import logging

import numpy as np
import torch.nn as nn
from torch import Tensor


# ========================================================================= #
# Utility Layers                                                            #
# ========================================================================= #

class Print(nn.Module):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.log = logging.getLogger(__name__)

    def forward(self, tensor):
        self.log.debug(self.layer, '|', tensor.shape, '->')
        output = self.layer.forward(tensor)
        self.log.debug(output.shape)
        return output

class BatchView(nn.Module):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, size):
        super().__init__()
        self.size = (-1, *size)

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


# ========================================================================= #
# Custom Base nn.Module                                                     #
# ========================================================================= #


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
        assert x.shape[1:] == self.x_shape, f'X shape mismatch. Required: {self.x_shape} Got: {x.shape[1:]}'
    def assert_z_valid(self, z):
        assert z.ndim == 2
        assert z.shape[1] == self.z_size, f'Z size mismatch. Required: {self.z_size} Got: {z.shape[1]}'
    def assert_lengths(self, x, z):
        assert len(x) == len(z)


# class BaseEncoderModule(BaseModule):
#     def forward(self, x) -> Tensor:
#         self.assert_x_valid(x)
#         # encode
#         z = self.encode(x)
#         # checks
#         self.assert_z_valid(z)
#         self.assert_lengths(x, z)
#         # return
#         return z
#
#     def encode(self, x) -> Tensor:
#         raise NotImplementedError


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
# END                                                                       #
# ========================================================================= #
