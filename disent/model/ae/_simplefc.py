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


# ========================================================================= #
# simple fully connected models                                             #
# ========================================================================= #


class EncoderSimpleFC(DisentEncoder):
    """
    Custom Fully Connected Encoder.
    """

    def __init__(self, x_shape=(3, 64, 64), h_size1=128, h_size2=128, z_size=6, z_multiplier=1):
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.x_size, out_features=h_size1), nn.ReLU(True),
            nn.Linear(in_features=h_size1,     out_features=h_size2), nn.ReLU(True),
            nn.Linear(in_features=h_size2,     out_features=self.z_total)
        )

    def encode(self, x) -> (Tensor, Tensor):
        return self.model(x)


class DecoderSimpleFC(DisentDecoder):
    """
    Custom Fully Connected Decoder.
    """

    def __init__(self, x_shape=(3, 64, 64), h_size1=128, h_size2=128, z_size=6, z_multiplier=1):
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)
        self.model = nn.Sequential(
            nn.Linear(in_features=self.z_size, out_features=h_size2), nn.ReLU(True),
            nn.Linear(in_features=h_size2,     out_features=h_size1), nn.ReLU(True),
            nn.Linear(in_features=h_size1,     out_features=self.x_size),
            nn.Unflatten(dim=1, unflattened_size=self.x_shape),
        )

    def decode(self, z):
        return self.model(z)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
