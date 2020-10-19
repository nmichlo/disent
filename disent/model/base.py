import logging
import numpy as np
import torch.nn as nn
from torch import Tensor

log = logging.getLogger(__name__)

# ========================================================================= #
# Utility Layers                                                            #
# ========================================================================= #

class Print(nn.Module):
    """From: https://github.com/1Konny/Beta-VAE/blob/master/model.py"""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, tensor):
        log.debug(self.layer, '|', tensor.shape, '->')
        output = self.layer.forward(tensor)
        log.debug(output.shape)
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


class BaseModule(DisentModule):

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        super().__init__()
        self._x_shape = x_shape
        self._x_size = int(np.prod(x_shape))
        self._z_size = z_size
        self._z_multiplier = z_multiplier

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def x_shape(self):
        return self._x_shape
    @property
    def x_size(self):
        return self._x_size
    @property
    def z_size(self):
        return self._z_size
    @property
    def z_multiplier(self):
        return self._z_multiplier
    @property
    def z_total(self):
        return self._z_size * self._z_multiplier

    def assert_x_valid(self, x):
        assert x.ndim == 4
        assert x.shape[1:] == self.x_shape, f'X shape mismatch. Required: {self.x_shape} Got: {x.shape[1:]}'
    def assert_z_valid(self, z):
        assert z.ndim == 2
        assert z.shape[1] == self.z_total, f'Z size mismatch. Required: {self.z_size} (x{self.z_multiplier}) = {self.z_total} Got: {z.shape[1]}'
    def assert_lengths(self, x, z):
        assert len(x) == len(z)


# ========================================================================= #
# Custom Base nn.Module                                                     #
# ========================================================================= #


class BaseEncoderModule(BaseModule):

    @final
    def forward(self, x) -> Tensor:
        """same as self.encode but with size checks"""
        self.assert_x_valid(x)
        # encode | p(z|x)
        # for a gaussian encoder, we treat z as concat(z_mean, z_logvar) where z_mean.shape == z_logvar.shape
        # ie. the first half of z is z_mean, the second half of z is z_logvar
        z = self.encode(x)
        # checks
        self.assert_z_valid(z)
        self.assert_lengths(x, z)
        # return
        return z

    def encode(self, x) -> Tensor:
        raise NotImplementedError


class BaseDecoderModule(BaseModule):
    
    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        assert z_multiplier == 1, 'decoder does not support z_multiplier != 1'
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

    @final
    def forward(self, z):
        """same as self.decode but with size checks"""
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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
