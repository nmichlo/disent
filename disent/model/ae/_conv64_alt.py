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

from torch import nn as nn, Tensor

from disent.model.base import BaseEncoderModule, BaseDecoderModule
from disent.model.common import Flatten3D, BatchView


def _make_activations(activation='relu', inplace=True, norm='instance', num_features: int = None, norm_pre_act=True):
    # get activation layer
    if activation == 'relu':
        a_layer = nn.ReLU(inplace=inplace)
    elif activation == 'leaky_relu':
        a_layer = nn.LeakyReLU(inplace=inplace)
    else:
        raise KeyError(f'invalid activation layer: {repr(activation)}')
    # get norm layer
    # https://www.programmersought.com/article/41731094913/
    # nn.BatchNorm2d
    # nn.InstanceNorm2d
    # nn.GroupNorm
    # nn.LayerNorm
    if norm == 'batch':
        n_layer = nn.BatchNorm2d(num_features=num_features)
    elif norm == 'instance':
        n_layer = nn.InstanceNorm2d(num_features=num_features)
    elif norm in (None, 'none'):
        n_layer = None
    else:
        raise KeyError(f'invalid norm layer: {repr(norm)}')
    # order layers
    layers = (n_layer, a_layer) if norm_pre_act else (a_layer, n_layer)
    # return layers
    return tuple(l for l in layers if l is not None)


# ========================================================================= #
# disentanglement_lib Conv models                                           #
# ========================================================================= #


class EncoderConv64Alt(BaseEncoderModule):
    """
    Reference Implementation:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    # TODO: verify, things have changed...
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1, activation='leaky_relu', norm='instance', norm_pre_act=True):
        """
        Convolutional encoder used in beta-VAE paper for the chairs data.
        Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl)
        """
        # checks
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        num_channels = x_shape[0]
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=4, stride=2, padding=2),
                *_make_activations(activation=activation, norm=norm, num_features=32, norm_pre_act=norm_pre_act),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2),
                *_make_activations(activation=activation, norm=norm, num_features=32, norm_pre_act=norm_pre_act),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=1),
                *_make_activations(activation=activation, norm=norm, num_features=64, norm_pre_act=norm_pre_act),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=1),
                *_make_activations(activation=activation, norm=norm, num_features=64, norm_pre_act=norm_pre_act),
            Flatten3D(),
            nn.Linear(1600, 256),
                *_make_activations(activation=activation, norm='none'),
            nn.Linear(256, self.z_total),
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderConv64Alt(BaseDecoderModule):
    """
    From:
    https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/shared/architectures.py
    # TODO: verify, things have changed...
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1, activation='leaky_relu', norm='instance', norm_pre_act=True):
        """
        Convolutional decoder used in beta-VAE paper for the chairs data.
        Based on row 3 of Table 1 on page 13 of "beta-VAE: Learning Basic Visual
        Concepts with a Constrained Variational BaseFramework"
        (https://openreview.net/forum?id=Sy2fzU9gl)
        """
        assert tuple(x_shape[1:]) == (64, 64), 'This model only works with image size 64x64.'
        num_channels = x_shape[0]
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            nn.Linear(self.z_size, 256),
                *_make_activations(activation=activation, norm='none'),
            nn.Linear(256, 1024),
                *_make_activations(activation=activation, norm='none'),
            BatchView([64, 4, 4]),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1),
                *_make_activations(activation=activation, norm=norm, num_features=64, norm_pre_act=norm_pre_act),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
                *_make_activations(activation=activation, norm=norm, num_features=32, norm_pre_act=norm_pre_act),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
                *_make_activations(activation=activation, norm=norm, num_features=32, norm_pre_act=norm_pre_act),
            nn.ConvTranspose2d(in_channels=32, out_channels=num_channels, kernel_size=4, stride=2, padding=1),
        )

    def decode(self, z) -> Tensor:
        return self.model(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
