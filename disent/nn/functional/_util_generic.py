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

from typing import Union

import numpy as np
import torch


# ========================================================================= #
# Generic Helper Functions                                                  #
# - torch, numpy, scalars                                                   #
# - eagerpy is one solution to including this ourselves                     #
# ========================================================================= #


TypeGenericTensor = Union[float, int, torch.Tensor, np.ndarray]
TypeGenericTorch   = Union[float, int, torch.Tensor]
TypeGenericNumpy   = Union[float, int, np.ndarray]


def generic_as_int32(input: TypeGenericTensor):
    if isinstance(input, (float, int)):
        return int(input)
    elif isinstance(input, torch.Tensor):
        return input.to(torch.int32)
    elif isinstance(input, np.ndarray):
        return input.astype(np.int32)
    else:
        raise TypeError(f'invalid type: {type(input)}')


def generic_max(input: TypeGenericTensor):
    if isinstance(input, (float, int)):
        return input
    elif isinstance(input, torch.Tensor):
        return input.max()
    elif isinstance(input, np.ndarray):
        return input.max()
    else:
        raise TypeError(f'invalid type: {type(input)}')


def generic_min(input: TypeGenericTensor):
    if isinstance(input, (float, int)):
        return input
    elif isinstance(input, torch.Tensor):
        return input.min()
    elif isinstance(input, np.ndarray):
        return input.min()
    else:
        raise TypeError(f'invalid type: {type(input)}')


def generic_shape(input: TypeGenericTensor):
    if isinstance(input, (float, int)):
        return ()
    elif isinstance(input, torch.Tensor):
        return input.shape
    elif isinstance(input, np.ndarray):
        return input.shape
    else:
        raise TypeError(f'invalid type: {type(input)}')


def generic_ndim(input: TypeGenericTensor):
    if isinstance(input, (float, int)):
        return 0
    elif isinstance(input, torch.Tensor):
        return input.ndim
    elif isinstance(input, np.ndarray):
        return input.ndim
    else:
        raise TypeError(f'invalid type: {type(input)}')


# ========================================================================= #
# end                                                                       #
# ========================================================================= #
