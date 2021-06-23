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

from torch import nn
from torch import Tensor

from disent.model import DisentDecoder
from disent.model import DisentEncoder
from disent.nn.activations import Swish
from disent.nn.modules import BatchView
from disent.nn.modules import Flatten3D


# ========================================================================= #
# disentanglement_lib Conv models                                           #
# ========================================================================= #


class EncoderConv64Ws(DisentEncoder):
    """
    Modified version of `EncoderConv64` with
    swish activations and scaled convolutions
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        # checks
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), 'This model only works with image size 64x64.'
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            self.Conv2d(in_channels=C,  out_channels=32, kernel_size=4, stride=2, padding=2), self.Activation(),
            self.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2), self.Activation(),
            self.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2, padding=1), self.Activation(),
            self.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=1), self.Activation(),
            Flatten3D(),
            self.Linear(in_features=1600, out_features=256), self.Activation(),
            self.Linear(in_features=256,  out_features=self.z_total),  # we combine the two networks in the reference implementation and use torch.chunk(2, dim=-1) to get mu & logvar
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)

    # COMPONENTS

    def Activation(self):
        return Swish()

    def Conv2d(self, **kwargs):
        from nfnets import ScaledStdConv2d
        # This gamma value is specific to swish. For details on how to
        # compute gamma, see appendix D of: https://arxiv.org/pdf/2101.08692.pdf
        return ScaledStdConv2d(**kwargs, gamma=1.7872)

    def Linear(self, **kwargs):
        return nn.Linear(**kwargs)


class DecoderConv64Ws(DisentDecoder):
    """
    Modified version of `DecoderConv64` with
    swish activations and scaled convolutions
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1):
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), 'This model only works with image size 64x64.'
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)

        self.model = nn.Sequential(
            self.Linear(in_features=self.z_size, out_features=256),  self.Activation(),
            self.Linear(in_features=256,         out_features=1024), self.Activation(),
            BatchView([64, 4, 4]),
            self.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1), self.Activation(),
            self.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), self.Activation(),
            self.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1), self.Activation(),
            self.ConvTranspose2d(in_channels=32, out_channels=C,  kernel_size=4, stride=2, padding=1),
        )

    def decode(self, z) -> Tensor:
        return self.model(z)

    # COMPONENTS

    def Activation(self):
        return Swish()

    def ConvTranspose2d(self, **kwargs):
        return nn.ConvTranspose2d(**kwargs)

    def Linear(self, **kwargs):
        return nn.Linear(**kwargs)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
