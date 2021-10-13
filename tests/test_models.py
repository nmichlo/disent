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
import pytest
import torch

from disent.model import AutoEncoder
from disent.model import DisentDecoder
from disent.model import DisentEncoder
from disent.model.ae import *


@pytest.mark.parametrize(['encoder_cls', 'decoder_cls'], [
    [EncoderConv64,     DecoderConv64],
    [EncoderConv64Norm, DecoderConv64Norm],
    [EncoderFC,         DecoderFC],
    [EncoderLinear, DecoderLinear],
])
def test_ae_models(encoder_cls: DisentEncoder, decoder_cls: DisentDecoder):
    x_shape, z_size = (3, 64, 64), 8
    # create model
    vae = AutoEncoder(
        encoder_cls(x_shape=x_shape, z_size=z_size, z_multiplier=2),
        decoder_cls(x_shape=x_shape, z_size=z_size, z_multiplier=1),
    )
    # feed forward
    with torch.no_grad():
        x = torch.randn(1, *x_shape, dtype=torch.float32)
        assert x.shape == (1, *x_shape)
        z0, z1 = vae.encode(x, chunk=True)
        assert z0.shape == (1, z_size)
        assert z1.shape == (1, z_size)
        y = vae.decode(z0)
        assert y.shape == (1, *x_shape)
