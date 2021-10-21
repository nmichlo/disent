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
# helper                                                                    #
# ========================================================================= #


def torch_normalize(tensor: torch.Tensor, dims=None, dtype=None):
    # get min & max values
    if dims is not None:
        m, M = tensor, tensor
        for dim in dims:
            m, M = m.min(dim=dim, keepdim=True).values, M.max(dim=dim, keepdim=True).values
    else:
        m, M = tensor.min(), tensor.max()
    # scale tensor
    return (tensor.to(dtype=dtype) - m) / (M - m)  # automatically converts to float32 if needed


# ========================================================================= #
# polyfill - in later versions of pytorch                                   #
# ========================================================================= #


def torch_nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    output = input.clone()
    if nan is not None:
        output[torch.isnan(input)] = nan
    if posinf is not None:
        output[input == np.inf] = posinf
    if neginf is not None:
        output[input == -np.inf] = neginf
    return output


# ========================================================================= #
# Torch Dim Helper                                                          #
# ========================================================================= #


def torch_unsqueeze_l(input: torch.Tensor, n: int):
    """
    Add n new axis to the left.

    eg. a tensor with shape (2, 3) passed to this function
        with n=2 will input in an output shape of (1, 1, 2, 3)
    """
    assert n >= 0, f'number of new axis cannot be less than zero, given: {repr(n)}'
    return input[((None,)*n) + (...,)]


def torch_unsqueeze_r(input: torch.Tensor, n: int):
    """
    Add n new axis to the right.

    eg. a tensor with shape (2, 3) passed to this function
        with n=2 will input in an output shape of (2, 3, 1, 1)
    """
    assert n >= 0, f'number of new axis cannot be less than zero, given: {repr(n)}'
    return input[(...,) + ((None,)*n)]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
