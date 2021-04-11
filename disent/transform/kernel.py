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
import torch
import numpy as np


# ========================================================================= #
# Augment                                                                   #
# ========================================================================= #


def get_kernel_size(sigma=1.0, truncate=4.0):
    # HOW SKLEARN CHOOSES FILTER SIZES:
    # sigma is the std, and truncate is the number of std, default for truncate is 4, default for sigma is 1
    # radius = int(truncate * sigma + 0.5)
    # size = 2*radius+1
    return int(truncate * sigma + 0.5) * 2 + 1


def make_gaussian_kernel(sigma=1.0, truncate=4.0, size=None, device=None, dtype=None):
    if size is None:
        size = get_kernel_size(sigma=sigma, truncate=truncate)
    # compute kernel
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    return torch.exp(-(x ** 2) / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma)


def make_gaussian_kernel_2d(sigma=1.0, truncate=4.0, size=None):
    if not isinstance(sigma, (tuple, list)): sigma = (sigma, sigma)
    if not isinstance(truncate, (tuple, list)): truncate = (truncate, truncate)
    if not isinstance(size, (tuple, list)): size = (size, size)
    # compute kernel
    kh = make_gaussian_kernel(sigma=sigma[0], truncate=truncate[0], size=size[0])
    kw = make_gaussian_kernel(sigma=sigma[1], truncate=truncate[1], size=size[1])
    return kh[:, None] * kw[None, :]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
