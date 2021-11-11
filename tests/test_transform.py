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

from disent.dataset.transform import FftGaussianBlur
from disent.dataset.transform._augment import _expand_to_min_max_tuples
from disent.nn.functional import torch_gaussian_kernel
from disent.nn.functional import torch_gaussian_kernel_2d


# ========================================================================= #
# TESTS                                                                     #
# ========================================================================= #


def test_gaussian_kernels():

    r = lambda *shape: torch.stack(torch.meshgrid(*((torch.arange(s) + 1) for s in shape)), dim=-1).max(dim=-1).values

    assert (9,)       == torch_gaussian_kernel(sigma=1.0,     truncate=4.0).shape
    assert (5, 41)    == torch_gaussian_kernel(sigma=r(5),    truncate=4.0).shape
    assert (5, 3, 41) == torch_gaussian_kernel(sigma=r(5, 3), truncate=4.0).shape
    assert (5, 7, 71) == torch_gaussian_kernel(sigma=r(5, 1), truncate=r(1, 7)).shape

    assert (9, 7)         == torch_gaussian_kernel_2d(sigma=1.0,     truncate=4.0, sigma_b=1.0,     truncate_b=3.0).shape
    assert (5, 41, 9)     == torch_gaussian_kernel_2d(sigma=r(5),    truncate=4.0, sigma_b=1.0,     truncate_b=4.0).shape
    assert (5, 7, 41, 57) == torch_gaussian_kernel_2d(sigma=r(5, 1), truncate=4.0, sigma_b=r(1, 7), truncate_b=4.0).shape


def test_fft_guassian_blur_sigmas():
    assert _expand_to_min_max_tuples(1.0)                      == ((1.0, 1.0), (1.0, 1.0))
    assert _expand_to_min_max_tuples([0.0, 1.0])               == ((0.0, 1.0), (0.0, 1.0))
    assert _expand_to_min_max_tuples([[0.0, 1.0], [1.0, 2.0]]) == ((0.0, 1.0), (1.0, 2.0))
    with pytest.raises(Exception):
        _expand_to_min_max_tuples([[0.0, 1.0], 1.0])
    with pytest.raises(Exception):
        _expand_to_min_max_tuples([0.0, [1.0, 2.0]])


def test_fft_guassian_blur():
    fn = FftGaussianBlur(sigma=1.0, truncate=3.0)
    fn(torch.randn(256, 3, 64, 64))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
