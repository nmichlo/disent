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

import numpy as np
import torch


# ========================================================================= #
# Discrete Cosine Transform                                                 #
# ========================================================================= #


def _flatten_dim_to_end(input, dim):
    # get shape
    s = input.shape
    n = s[dim]
    # do operation
    x = torch.moveaxis(input, dim, -1)
    x = x.reshape(-1, n)
    return x, s, n


def _unflatten_dim_to_end(input, dim, shape):
    # get intermediate shape
    s = list(shape)
    s.append(s.pop(dim))
    # undo operation
    x = input.reshape(*s)
    x = torch.moveaxis(x, -1, dim)
    return x


def torch_dct(x, dim=-1):
    """
    Discrete Cosine Transform (DCT) Type II
    """
    x, x_shape, n = _flatten_dim_to_end(x, dim=dim)
    if n % 2 != 0:
        raise ValueError(f'dct does not support odd sized dimension! trying to compute dct over dimension: {dim} of tensor with shape: {x_shape}')

    # concatenate even and odd offsets
    v_evn = x[:, 0::2]
    v_odd = x[:, 1::2].flip([1])
    v = torch.cat([v_evn, v_odd], dim=-1)

    # fast fourier transform
    fft = torch.fft.fft(v)

    # compute real & imaginary forward weights
    k = torch.arange(n, dtype=x.dtype, device=x.device) * (-np.pi / (2 * n))
    k = k[None, :]
    wr = torch.cos(k) * 2
    wi = torch.sin(k) * 2

    # compute dct
    dct = torch.real(fft) * wr - torch.imag(fft) * wi

    # restore shape
    return _unflatten_dim_to_end(dct, dim, x_shape)


def torch_idct(dct, dim=-1):
    """
    Inverse Discrete Cosine Transform (Inverse DCT) Type III
    """
    dct, dct_shape, n = _flatten_dim_to_end(dct, dim=dim)
    if n % 2 != 0:
        raise ValueError(f'idct does not support odd sized dimension! trying to compute idct over dimension: {dim} of tensor with shape: {dct_shape}')

    # compute real & imaginary backward weights
    k = torch.arange(n, dtype=dct.dtype, device=dct.device) * (np.pi / (2 * n))
    k = k[None, :]
    wr = torch.cos(k) / 2
    wi = torch.sin(k) / 2

    dct_real = dct
    dct_imag = torch.cat([0*dct_real[:, :1], -dct_real[:, 1:].flip([1])], dim=-1)

    fft_r = dct_real * wr - dct_imag * wi
    fft_i = dct_real * wi + dct_imag * wr
    # to complex number
    fft = torch.view_as_complex(torch.stack([fft_r, fft_i], dim=-1))

    # inverse fast fourier transform
    v = torch.fft.ifft(fft)
    v = torch.real(v)

    # undo even and odd offsets
    x = torch.zeros_like(dct)
    x[:, 0::2]  = v[:, :(n+1)//2]  # (N+1)//2 == N-(N//2)
    x[:, 1::2] += v[:, (n+0)//2:].flip([1])

    # restore shape
    return _unflatten_dim_to_end(x, dim, dct_shape)


def torch_dct2(x, dim1=-1, dim2=-2):
    d = torch_dct(x, dim=dim1)
    d = torch_dct(d, dim=dim2)
    return d


def torch_idct2(d, dim1=-1, dim2=-2):
    x = torch_idct(d, dim=dim2)
    x = torch_idct(x, dim=dim1)
    return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
