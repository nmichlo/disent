#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import logging
from typing import final

import numpy as np
from torch import Tensor

from disent.util import DisentModule


log = logging.getLogger(__name__)


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
