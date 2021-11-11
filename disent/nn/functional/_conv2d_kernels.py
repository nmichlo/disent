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

from disent.nn.functional._other import torch_unsqueeze_l
from disent.nn.functional._other import torch_unsqueeze_r
from disent.nn.functional._util_generic import generic_as_int32
from disent.nn.functional._util_generic import generic_max
from disent.nn.functional._util_generic import TypeGenericTensor
from disent.nn.functional._util_generic import TypeGenericTorch


# ========================================================================= #
# Kernels                                                                   #
# ========================================================================= #


# TODO: replace with meshgrid based functions from experiment/exp/06_metric
#       these are more intuitive and flexible


def get_kernel_size(sigma: TypeGenericTensor = 1.0, truncate: TypeGenericTensor = 4.0):
    """
    This is how sklearn chooses kernel sizes.
    - sigma is the standard deviation, and truncate is the number of deviations away to truncate
    - our version broadcasts sigma and truncate together, returning the max kernel size needed over all values
    """
    # compute radius
    radius = generic_as_int32(truncate * sigma + 0.5)
    # get maximum value
    radius = int(generic_max(radius))
    # compute diameter
    return 2 * radius + 1


def torch_gaussian_kernel(
    sigma: TypeGenericTorch = 1.0, truncate: TypeGenericTorch = 4.0, size: int = None,
    dtype=torch.float32, device=None,
):
    # broadcast tensors together -- data may reference single memory locations
    sigma = torch.as_tensor(sigma, dtype=dtype, device=device)
    truncate = torch.as_tensor(truncate, dtype=dtype, device=device)
    sigma, truncate = torch.broadcast_tensors(sigma, truncate)
    # compute default size
    if size is None:
        size: int = get_kernel_size(sigma=sigma, truncate=truncate)
    # compute kernel
    x = torch.arange(size, dtype=sigma.dtype, device=sigma.device) - (size - 1) / 2
    # pad tensors correctly
    x = torch_unsqueeze_l(x, n=sigma.ndim)
    s = torch_unsqueeze_r(sigma, n=1)
    # compute
    return torch.exp(-(x ** 2) / (2 * s ** 2)) / (np.sqrt(2 * np.pi) * s)


def torch_gaussian_kernel_2d(
    sigma: TypeGenericTorch = 1.0, truncate: TypeGenericTorch = 4.0, size: int = None,
    sigma_b: TypeGenericTorch = None, truncate_b: TypeGenericTorch = None, size_b: int = None,
    dtype=torch.float32, device=None,
):
    # set default values
    if sigma_b is None: sigma_b = sigma
    if truncate_b is None: truncate_b = truncate
    if size_b is None: size_b = size
    # compute kernel
    kh = torch_gaussian_kernel(sigma=sigma,   truncate=truncate,   size=size,   dtype=dtype, device=device)
    kw = torch_gaussian_kernel(sigma=sigma_b, truncate=truncate_b, size=size_b, dtype=dtype, device=device)
    return kh[..., :, None] * kw[..., None, :]


def torch_box_kernel(radius: TypeGenericTorch = 1, dtype=torch.float32, device=None):
    radius = torch.abs(torch.as_tensor(radius, device=device))
    assert radius.dtype in {torch.int32, torch.int64}, f'box kernel radius must be of integer type: {radius.dtype}'
    # box kernel values
    radius_max = radius.max()
    crange = torch.abs(torch.arange(radius_max * 2 + 1, dtype=dtype, device=device) - radius_max)
    # pad everything
    radius = radius[..., None]
    crange = crange[None, ...]
    # compute box kernel
    kernel = (crange <= radius).to(dtype) / (radius * 2 + 1)
    # done!
    return kernel


def torch_box_kernel_2d(
    radius: TypeGenericTorch = 1,
    radius_b: TypeGenericTorch = None,
    dtype=torch.float32, device=None
):
    # set default values
    if radius_b is None: radius_b = radius
    # compute kernel
    kh = torch_box_kernel(radius=radius,   dtype=dtype, device=device)
    kw = torch_box_kernel(radius=radius_b, dtype=dtype, device=device)
    return kh[..., :, None] * kw[..., None, :]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
