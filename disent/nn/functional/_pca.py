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
# PCA                                                                       #
# ========================================================================= #


def torch_pca_eig(X, center=True, scale=False):
    """
    perform PCA over X
    - X is of size (num_points, vec_size)

    NOTE: unlike PCA_svd, the number of vectors/values returned is always: vec_size
    """
    n, _ = X.shape
    # center points along axes
    if center:
        X = X - X.mean(dim=0)
    # compute covariance -- TODO: optimise this line
    covariance = (1 / (n-1)) * torch.mm(X.T, X)
    if scale:
        scaling = torch.sqrt(1 / torch.diagonal(covariance))
        covariance = torch.mm(torch.diagflat(scaling), covariance)
    # compute eigen values and eigen vectors
    eigenvalues, eigenvectors = torch.eig(covariance, True)
    # sort components by decreasing variance
    components = eigenvectors.T
    explained_variance = eigenvalues[:, 0]
    idxs = torch.argsort(explained_variance, descending=True)
    return components[idxs], explained_variance[idxs]


def torch_pca_svd(X, center=True):
    """
    perform PCA over X
    - X is of size (num_points, vec_size)

    NOTE: unlike PCA_eig, the number of vectors/values returned is: min(num_points, vec_size)
    """
    n, _ = X.shape
    # center points along axes
    if center:
        X = X - X.mean(dim=0)
    # perform singular value decomposition
    u, s, v = torch.svd(X)
    # sort components by decreasing variance
    # these are already sorted?
    components = v.T
    explained_variance = torch.mul(s, s) / (n-1)
    return components, explained_variance


def torch_pca(X, center=True, mode='svd'):
    if mode == 'svd':
        return torch_pca_svd(X, center=center)
    elif mode == 'eig':
        return torch_pca_eig(X, center=center, scale=False)
    else:
        raise KeyError(f'invalid torch_pca mode: {repr(mode)}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
