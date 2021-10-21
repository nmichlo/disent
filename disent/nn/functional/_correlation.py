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

    TODO: check, be careful of repeated values, this might not give the correct input?
    """
    rs = torch.argsort(xs, dim=0, descending=False)
    return torch_corr_matrix(rs.to(xs.dtype))


# aliases
torch_pearsons_corr_matrix = torch_corr_matrix
torch_spearmans_corr_matrix = torch_rank_corr_matrix


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
