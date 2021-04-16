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
from functools import lru_cache

from typing import Tuple
from typing import Union

import numpy as np
import torch


# ========================================================================= #
# helper functions for moving dimensions                                    #
# ========================================================================= #


@lru_cache(maxsize=32)
def _get_2d_reshape_info(shape: Tuple[int, ...], dims: Union[int, Tuple[int, ...]] = -1):
    if isinstance(dims, int):
        dims = (dims,)
    # number of dimensions & remove negatives
    ndim = len(shape)
    dims = tuple((ndim + d) if d < 0 else d for d in dims)
    # check that we have at least 2 dims & that all values are valid
    assert ndim >= 2
    assert all(0 <= d < ndim for d in dims)
    # get dims
    dims_X = set(dims)
    dims_B = set(range(ndim)) - dims_X
    # sort dims
    dims_X = sorted(dims_X)
    dims_B = sorted(dims_B)
    # compute shape
    shape = np.array(shape)
    size_B = int(np.prod(shape[dims_B]))
    size_X = int(np.prod(shape[dims_X]))
    # variables
    move_end_dims = dims_X[::-1]
    reshape_size = (size_B, size_X)
    # sort dims
    return move_end_dims, reshape_size


def torch_dims_at_end_2d(tensor: torch.Tensor, dims: Union[int, Tuple[int, ...]] = -1):
    # get dim info
    move_end_dims, reshape_size = _get_2d_reshape_info(tensor.shape, dims=dims)
    # move all axes
    for d in move_end_dims:
        tensor = torch.moveaxis(tensor, d, -1)
    # reshape
    return torch.reshape(tensor, reshape_size)


# ========================================================================= #
# Improved differentiable torchsort functions                               #
# ========================================================================= #


def torch_soft_sort(
    tensor: torch.Tensor,
    dims: Union[int, Tuple[int, ...]] = -1,
    regularization='l2',
    regularization_strength=1.0,
):
    # TODO: convert back to original shape?
    # we import it locally so that we don't have to install this
    import torchsort
    # reorder the dimensions
    tensor = torch_dims_at_end_2d(tensor, dims=dims)
    # sort the last dimension of the 2D tensors
    return torchsort.soft_sort(tensor, regularization=regularization, regularization_strength=regularization_strength)


def torch_soft_rank(
    tensor: torch.Tensor,
    dims: Union[int, Tuple[int, ...]] = -1,
    regularization='l2',
    regularization_strength=1.0,
):
    # TODO: convert back to original shape?
    # we import it locally so that we don't have to install this
    import torchsort
    # reorder the dimensions
    tensor = torch_dims_at_end_2d(tensor, dims=dims)
    # sort the last dimension of the 2D tensors
    return torchsort.soft_rank(tensor, regularization=regularization, regularization_strength=regularization_strength)


# ========================================================================= #
# Differentiable Spearman rank correlation                                  #
# ========================================================================= #


def multi_spearman_rank_loss(
    pred: torch.Tensor,
    targ: torch.Tensor,
    dims: Union[int, Tuple[int, ...]] = -1,
    reduction='mean',
    regularization='l2',
    regularization_strength=1.0,
    nan_to_num=False,
) -> torch.Tensor:
    """
    Like spearman_rank_loss but computes the loss over the specified
    dims, taking the average over the remaining unspecified dims.

    eg. given a tensor of shape (B, C, H, W) and specifying dims=(1, 3), the
        loss is computed over the C & W dims, by internally moving the dimensions
        to the end (B, H, C, W) and the reshaping the array to (B*H, C*W) before
        passing the array to spearman_rank_loss and returning the mean loss.
    """
    assert pred.shape == targ.shape
    pred = torch_dims_at_end_2d(pred.shape, dims=dims)
    targ = torch_dims_at_end_2d(targ.shape, dims=dims)
    # compute
    assert reduction == 'mean', 'only supports reduction="mean"'
    return spearman_rank_loss(pred=pred, targ=targ, reduction=reduction, regularization=regularization, regularization_strength=regularization_strength, nan_to_num=nan_to_num)


def spearman_rank_loss(
    pred: torch.Tensor,
    targ: torch.Tensor,
    reduction='mean',
    regularization='l2',
    regularization_strength=1.0,
    nan_to_num=False,
):
    """
    Compute the spearman correlation coefficient.
    - Supports 1D and 2D tensors.
    - For a 2D tensor of shape (B, N), the spearman correlation
      coefficient is computed over the last axis, with
      the mean taken over the B resulting values.
    - If the tensor is 1D of shape (N,) it is converted
      automatically to a 2D tensor of shape (1, N)

    Adapted from: https://github.com/teddykoker/torchsort
    """
    # TODO: I think something is wrong with this...
    # we import it locally so that we don't have to install this
    import torchsort
    # add missing dim
    if pred.ndim == 1:
        pred, targ = pred.reshape(1, -1), targ.reshape(1, -1)
    assert pred.shape == targ.shape
    assert pred.ndim == 2
    # sort the last dimension of the 2D tensors
    assert regularization == 'l2', 'Only l2 regularization is currently supported for torchsort, others can result in memory leaks. See the torchsort github page for the bug report.'
    pred = torchsort.soft_rank(pred, regularization=regularization, regularization_strength=regularization_strength)
    targ = torchsort.soft_rank(targ, regularization=regularization, regularization_strength=regularization_strength)
    # compute individual losses
    # TODO: this can result in nan values, what to do then?
    pred = pred - pred.mean(dim=-1, keepdim=True)
    pred = pred / pred.norm(dim=-1, keepdim=True)
    targ = targ - targ.mean(dim=-1, keepdim=True)
    targ = targ / targ.norm(dim=-1, keepdim=True)
    # replace nan values
    if nan_to_num:
        pred = torch.nan_to_num(pred, nan=0.0)
        targ = torch.nan_to_num(targ, nan=0.0)
    # compute the final loss
    loss = (pred * targ).sum(dim=-1)
    # reduce the loss
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss
    else:
        raise KeyError(f'Invalid reduction mode: {repr(reduction)}')


# ========================================================================= #
# end                                                                       #
# ========================================================================= #

