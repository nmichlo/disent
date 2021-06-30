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

import warnings
from typing import Tuple

from torch import nn
from torch import Tensor

from disent.model import DisentDecoder
from disent.model import DisentEncoder
from disent.nn.activations import Swish


# ========================================================================= #
# disentanglement_lib Conv models                                           #
# ========================================================================= #


class EncoderConv64Norm(DisentEncoder):
    """
    Modified version of `EncoderConv64` with
    selectable activations and norm layers
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1, activation='leaky_relu', norm='layer', norm_pre_act=True):
        # checks
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), 'This model only works with image size 64x64.'
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        # warnings
        warnings.warn(f'this model: {self.__class__.__name__} is non-standard in VAE research!')

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=C,  out_channels=32, kernel_size=4, stride=2, padding=2), *_make_activations(activation=activation, norm=norm, shape=(32, 33, 33), norm_pre_act=norm_pre_act),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2), *_make_activations(activation=activation, norm=norm, shape=(32, 17, 17), norm_pre_act=norm_pre_act),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=1), *_make_activations(activation=activation, norm=norm, shape=(64,  9,  9), norm_pre_act=norm_pre_act),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=1), *_make_activations(activation=activation, norm=norm, shape=(64,  5,  5), norm_pre_act=norm_pre_act),
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=256), *_make_activations(activation=activation, norm='none'),
            nn.Linear(in_features=256,  out_features=self.z_total),
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderConv64Norm(DisentDecoder):
    """
    Modified version of `DecoderConv64` with
    selectable activations and norm layers
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1, activation='leaky_relu', norm='layer', norm_pre_act=True):
        # checks
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), 'This model only works with image size 64x64.'
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        # warnings
        warnings.warn(f'this model: {self.__class__.__name__} is non-standard in VAE research!')

        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=256),  *_make_activations(activation=activation, norm='none'),
            nn.Linear(in_features=256,         out_features=1024), *_make_activations(activation=activation, norm='none'),
            nn.Unflatten(dim=1, unflattened_size=[64, 4, 4]),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1), *_make_activations(activation=activation, norm=norm, shape=(64,  8,  8), norm_pre_act=norm_pre_act),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), *_make_activations(activation=activation, norm=norm, shape=(32, 16, 16), norm_pre_act=norm_pre_act),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1), *_make_activations(activation=activation, norm=norm, shape=(32, 32, 32), norm_pre_act=norm_pre_act),
            nn.ConvTranspose2d(in_channels=32, out_channels=C,  kernel_size=4, stride=2, padding=1),
        )

    def decode(self, z) -> Tensor:
        return self.model(z)


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def _make_activations(activation='relu', inplace=True, norm='layer', shape: Tuple[int, ...] = None, norm_pre_act=True):
    # get activation layer
    if activation == 'relu':
        a_layer = nn.ReLU(inplace=inplace)
    elif activation == 'leaky_relu':
        a_layer = nn.LeakyReLU(inplace=inplace)
    elif activation == 'swish':
        a_layer = Swish()
    else:
        raise KeyError(f'invalid activation layer: {repr(activation)}')

    # get norm layer
    # https://www.programmersought.com/article/41731094913/
    # TODO: nn.GroupNorm

    if norm in (None, 'none'):
        n_layer = None
    else:
        C, H, W = shape
        if norm == 'batch':
            n_layer = nn.BatchNorm2d(num_features=C)
        elif norm == 'instance':
            n_layer = nn.InstanceNorm2d(num_features=C)
        elif norm == 'layer':
            n_layer = nn.LayerNorm(normalized_shape=[H, W])
        elif norm == 'layer_chn':
            n_layer = nn.LayerNorm(normalized_shape=[C, H, W])
        else:
            raise KeyError(f'invalid norm layer: {repr(norm)}')
    # order layers
    layers = (n_layer, a_layer) if norm_pre_act else (a_layer, n_layer)
    # return layers
    return tuple(l for l in layers if l is not None)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
