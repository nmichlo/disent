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

_P_NORM_MAP = {
    'inf':       _POS_INF,
    'maximum':   _POS_INF,
    'euclidean': 2,
    'manhattan': 1,
    # p < 1 is not a valid norm! but allow it if we set `unbounded_p = True`
    'hamming':   0,
    'minimum':   _NEG_INF,
    '-inf':      _NEG_INF,
}


# ========================================================================= #
# p-Norm Functions                                                          #
# -- very similar to the generalised mean                                   #
# ========================================================================= #


def torch_dist(xs: torch.Tensor, dim: _DimTypeHint = -1, p: Union[float, str] = 1, keepdim: bool = False):
    """
    Like torch_norm, but allows arbitrary p values that may
    result in the returned values not being a valid norm.
    - norm's require p >= 1
    """
    if isinstance(p, str):
        p = _P_NORM_MAP[p]
    # get absolute values
    xs = torch.abs(xs)
    # compute the specific extreme cases
    # -- its kind of odd that the p-norm and generalised mean converge to the
    #    same values, just from different directions!
    if p == _POS_INF:
        return torch.amax(xs, dim=dim, keepdim=keepdim)
    elif p == _NEG_INF:
        return torch.amin(xs, dim=dim, keepdim=keepdim)
    # get the dimensions
    if dim is None:
        dim = list(range(xs.ndim))
    # warn if the type is wrong
    if p != 1:
        if xs.dtype != torch.float64:
            warnings.warn(f'Input tensor to p-norm might not have the required precision, type is {xs.dtype} not {torch.float64}.')
    # compute the specific cases
    if p == 0:
        # hamming distance -- number of non-zero entries of the vector
        return torch.sum(xs != 0, dim=dim, keepdim=keepdim, dtype=xs.dtype)
    if p == 1:
        # manhattan distance
        return torch.sum(xs, dim=dim, keepdim=keepdim)
    else:
        # p-norm (if p==2, then euclidean distance)
        return torch.sum(xs ** p, dim=dim, keepdim=keepdim) ** (1/p)


def torch_norm(xs: torch.Tensor, dim: _DimTypeHint = -1, p: Union[float, str] = 1, keepdim: bool = False):
    """
    Compute the generalised p-norm over the given dimension of a vector!
    - p values must be >= 1

    Closely related to the generalised mean.
        - p-norm:          (sum(|x|^p)) ^ (1/p)
        - gen-mean:  (1/n * sum(x ^ p)) ^ (1/p)
    """
    if isinstance(p, str):
        p = _P_NORM_MAP[p]
    # check values
    if p < 1:
        raise ValueError(f'p-norm cannot have a p value less than 1, got: {repr(p)}, to bypass this error set `unbounded_p=True`.')
    # return norm
    return torch_dist(xs=xs, dim=dim, p=p, keepdim=keepdim)


def torch_norm_euclidean(xs, dim: _DimTypeHint = -1, keepdim: bool = False):
    return torch_dist(xs, dim=dim, p='euclidean', keepdim=keepdim)


def torch_norm_manhattan(xs, dim: _DimTypeHint = -1, keepdim: bool = False):
    return torch_dist(xs, dim=dim, p='manhattan', keepdim=keepdim)


def torch_dist_hamming(xs, dim: _DimTypeHint = -1, keepdim: bool = False):
    return torch_dist(xs, dim=dim, p='hamming', keepdim=keepdim)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
