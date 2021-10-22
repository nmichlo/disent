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

import warnings
from typing import List
from typing import Optional
from typing import Union

import torch


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


_DimTypeHint = Optional[Union[int, List[int]]]

_POS_INF = float('inf')
_NEG_INF = float('-inf')

_GENERALIZED_MEAN_MAP = {
    'maximum':   _POS_INF,
    'quadratic':  2,
    'arithmetic': 1,
    'geometric':  0,
    'harmonic':  -1,
    'minimum':   _NEG_INF,
}


# ========================================================================= #
# Generalized Mean Functions                                                #
# ========================================================================= #


def torch_mean_generalized(xs: torch.Tensor, dim: _DimTypeHint = None, p: Union[int, str] = 1, keepdim: bool = False):
    """
    Compute the generalised mean.
    - p is the power

    harmonic mean ≤ geometric mean ≤ arithmetic mean
    - If values have the same units: Use the arithmetic mean.
    - If values have differing units: Use the geometric mean.
    - If values are rates: Use the harmonic mean.
    """
    if isinstance(p, str):
        p = _GENERALIZED_MEAN_MAP[p]
    # compute the specific extreme cases
    if p == _POS_INF:
        return torch.max(xs, dim=dim, keepdim=keepdim).values if (dim is not None) else torch.max(xs, keepdim=keepdim)
    elif p == _NEG_INF:
        return torch.min(xs, dim=dim, keepdim=keepdim).values if (dim is not None) else torch.min(xs, keepdim=keepdim)
    # compute the number of elements being averaged
    if dim is None:
        dim = list(range(xs.ndim))
    n = torch.prod(torch.as_tensor(xs.shape)[dim])
    # warn if the type is wrong
    if p != 1:
        if xs.dtype != torch.float64:
            warnings.warn(f'Input tensor to generalised mean might not have the required precision, type is {xs.dtype} not {torch.float64}.')
    # compute the specific cases
    if p == 0:
        # geometric mean
        # orig numerically unstable: torch.prod(xs, dim=dim) ** (1 / n)
        return torch.exp((1 / n) * torch.sum(torch.log(xs), dim=dim, keepdim=keepdim))
    elif p == 1:
        # arithmetic mean
        return torch.mean(xs, dim=dim, keepdim=keepdim)
    else:
        # generalised mean
        return ((1/n) * torch.sum(xs ** p, dim=dim, keepdim=keepdim)) ** (1/p)


def torch_mean_quadratic(xs, dim: _DimTypeHint = None, keepdim: bool = False):
    return torch_mean_generalized(xs, dim=dim, p='quadratic', keepdim=keepdim)


def torch_mean_geometric(xs, dim: _DimTypeHint = None, keepdim: bool = False):
    return torch_mean_generalized(xs, dim=dim, p='geometric', keepdim=keepdim)


def torch_mean_harmonic(xs, dim: _DimTypeHint = None, keepdim: bool = False):
    return torch_mean_generalized(xs, dim=dim, p='harmonic', keepdim=keepdim)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
