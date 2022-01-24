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
    'maximum':   _POS_INF,
    'euclidean':  2,
    'manhattan': 1,
    # p < 1 is not a valid norm! but allow it if we set `unbounded_p = True`
    'hamming': 0,
    'minimum':   _NEG_INF,
}


# ========================================================================= #
# p-Norm Functions                                                          #
# -- very similar to the generalised mean                                   #
# ========================================================================= #


def torch_norm_p(xs: torch.Tensor, dim: _DimTypeHint = None, p: Union[int, str] = 1, keepdim: bool = False, unbounded_p: bool = False):
    """
    Compute the generalised p-norm over the given dimension of a vector!
        - If `unbounded_p=True` then allow p values that are less than 1

    Closely related to the generalised mean.
        - p-norm:          (sum(|x|^p)) ^ (1/p)
        - gen-mean:  (1/n * sum(x ^ p)) ^ (1/p)
    """
    if isinstance(p, str):
        p = _P_NORM_MAP[p]
    # check values
    if not unbounded_p:
        if p < 1:
            raise ValueError(f'p-norm cannot have a p value less than 1, got: {repr(p)}, to bypass this error set `unbounded_p=True`.')
    # get absolute values
    xs = torch.abs(xs)
    # compute the specific extreme cases
    # -- its kind of odd that the p-norm and generalised mean converge to the
    #    same values, just from different directions!
    if p == _POS_INF:
        return torch.max(xs, dim=dim, keepdim=keepdim).values if (dim is not None) else torch.max(xs, keepdim=keepdim)
    elif p == _NEG_INF:
        return torch.min(xs, dim=dim, keepdim=keepdim).values if (dim is not None) else torch.min(xs, keepdim=keepdim)
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


def torch_norm_p_unbounded(xs: torch.Tensor, dim: _DimTypeHint = None, p: Union[int, str] = 1, keepdim: bool = False):
    return torch_norm_p(xs=xs, dim=dim, p=p, keepdim=keepdim, unbounded_p=True)


def torch_norm_euclidean(xs, dim: _DimTypeHint = None, keepdim: bool = False):
    return torch_norm_p(xs, dim=dim, p='euclidean', keepdim=keepdim, unbounded_p=False)


def torch_norm_manhattan(xs, dim: _DimTypeHint = None, keepdim: bool = False):
    return torch_norm_p(xs, dim=dim, p='manhattan', keepdim=keepdim, unbounded_p=False)


def torch_norm_hamming(xs, dim: _DimTypeHint = None, keepdim: bool = False):
    return torch_norm_p(xs, dim=dim, p='hamming', keepdim=keepdim, unbounded_p=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
