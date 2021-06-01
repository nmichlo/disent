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

from disent.nn.modules import DisentModule


log = logging.getLogger(__name__)


# ========================================================================= #
# Custom Base Module Involving Inputs & Representations                     #
# ========================================================================= #


class DisentLatentsBase(DisentModule):

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
# Base Encoder & Base Decoder                                               #
# ========================================================================= #


class DisentEncoder(DisentLatentsBase):

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


class DisentDecoder(DisentLatentsBase):

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
# Auto-Encoder Wrapper                                                      #
# ========================================================================= #


class AutoEncoder(DisentLatentsBase):

    def __init__(self, encoder: DisentEncoder, decoder: DisentDecoder):
        assert isinstance(encoder, DisentEncoder)
        assert isinstance(decoder, DisentDecoder)
        # check sizes
        assert encoder.x_shape == decoder.x_shape, 'x_shape mismatch'
        assert encoder.x_size == decoder.x_size, 'x_size mismatch - this should never happen if x_shape matches'
        assert encoder.z_size == decoder.z_size, 'z_size mismatch'
        # initialise
        super().__init__(x_shape=decoder.x_shape, z_size=decoder.z_size, z_multiplier=encoder.z_multiplier)
        # assign
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, x):
        raise RuntimeError('This has been disabled')

    def encode(self, x):
        z_raw = self._encoder(x)
        # extract components if necessary
        if self._z_multiplier == 1:
            return z_raw
        elif self.z_multiplier == 2:
            return z_raw[..., :self.z_size], z_raw[..., self.z_size:]
        else:
            raise KeyError(f'z_multiplier={self.z_multiplier} is unsupported')

    def decode(self, z: Tensor) -> Tensor:
        """
        decode the given representation.
        the returned tensor does not have an activation applied to it!
        """
        return self._decoder(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

