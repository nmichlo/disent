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
from typing import List
from typing import Optional
from typing import Union

import logging
import numpy as np
import torch


log = logging.getLogger(__name__)


# ========================================================================= #
# pytorch math correlation functions                                        #
# ========================================================================= #


def torch_cov_matrix(xs: torch.Tensor):
    """
    Calculate the covariance matrix of multiple samples (N) of random vectors of size (X)
    https://en.wikipedia.org/wiki/Covariance_matrix
    - The input shape is: (N, X)
    - The output shape is: (X, X)

    This should be the same as:
        np.cov(xs, rowvar=False, ddof=0)
    """
    # NOTE:
    #   torch.mm is strict matrix multiplication
    #   however if we multiply arrays with broadcasting:
    #   size(3, 1) * size(1, 2) -> size(3, 2)  # broadcast, not matmul
    #   size(1, 3) * size(2, 1) -> size(2, 3)  # broadcast, not matmul
    # CHECK:
    assert xs.ndim == 2  # (N, X)
    Rxx = torch.mean(xs[:, :, None] * xs[:, None, :], dim=0)  # (X, X)
    ux = torch.mean(xs, dim=0)  # (X,)
    Kxx = Rxx - (ux[:, None] * ux[None, :])  # (X, X)
    return Kxx


def torch_corr_matrix(xs: torch.Tensor):
    """
    Calculate the pearson's correlation matrix of multiple samples (N) of random vectors of size (X)
    https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    https://en.wikipedia.org/wiki/Covariance_matrix
    - The input shape is: (N, X)
    - The output shape is: (X, X)

    This should be the same as:
        np.corrcoef(xs, rowvar=False, ddof=0)
    """
    Kxx = torch_cov_matrix(xs)
    diag_Kxx = torch.rsqrt(torch.diagonal(Kxx))
    corr = Kxx * (diag_Kxx[:, None] * diag_Kxx[None, :])
    return corr


def torch_rank_corr_matrix(xs: torch.Tensor):
    """
    Calculate the spearman's rank correlation matrix of multiple samples (N) of random vectors of size (X)
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient
    - The input shape is: (N, X)
    - The output shape is: (X, X)

    Pearson's correlation measures linear relationships
    Spearman's correlation measures monotonic relationships (whether linear or not)
    - defined in terms of the pearson's correlation matrix of the rank variables

    TODO: check, be careful of repeated values, this might not give the correct result?
    """
    rs = torch.argsort(xs, dim=0, descending=False)
    return torch_corr_matrix(rs.to(xs.dtype))


# aliases
torch_pearsons_corr_matrix = torch_corr_matrix
torch_spearmans_corr_matrix = torch_rank_corr_matrix


# ========================================================================= #
# pytorch math helper functions                                             #
# ========================================================================= #


def torch_tril_mean(mat: torch.Tensor, diagonal=-1):
    """
    compute the mean of the lower triangular matrix.
    """
    # checks
    N, M = mat.shape
    assert N == M
    assert diagonal == -1
    # compute
    n = (N*(N-1))/2
    mean = torch.tril(mat, diagonal=diagonal).sum() / n
    # done
    return mean


# ========================================================================= #
# pytorch mean functions                                                    #
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


def torch_mean_generalized(xs: torch.Tensor, dim: _DimTypeHint = None, p: Union[int, str] = 1):
    """
    Generalised Mean
    - is this implementation actually correct?
    """
    if isinstance(p, str):
        p = _GENERALIZED_MEAN_MAP[p]
    # warn if the type is wrong
    if xs.dtype != torch.float64:
        log.warning(f'Input tensor to generalised mean might not have the required precision, type is {xs.dtype} not {torch.float64}.')
    # compute the specific extreme cases
    if p == _POS_INF:
        return torch.max(xs, dim=dim).values
    elif p == _NEG_INF:
        return torch.min(xs, dim=dim).values
    # compute the number of elements being averaged
    if dim is None:
        dim = list(range(xs.ndim))
    n = torch.prod(torch.as_tensor(xs.shape)[dim])
    # compute the specific cases
    if p == 0:
        # geometric mean
        # orig numerically unstable: torch.prod(xs, dim=dim) ** (1 / n)
        return torch.exp((1 / n) * torch.sum(torch.log(xs), dim=dim))
    elif p == 1:
        # arithmetic mean
        return torch.mean(xs, dim=dim)
    else:
        # generalised mean
        return ((1/n) * torch.sum(xs ** p, dim=dim)) ** (1/p)


def torch_mean_quadratic(xs, dim: _DimTypeHint = None):
    return torch_mean_generalized(xs, dim=dim, p='quadratic')


def torch_mean_geometric(xs, dim: _DimTypeHint = None):
    return torch_mean_generalized(xs, dim=dim, p='geometric')


def torch_mean_harmonic(xs, dim: _DimTypeHint = None):
    return torch_mean_generalized(xs, dim=dim, p='harmonic')


# ========================================================================= #
# end                                                                       #
# ========================================================================= #
