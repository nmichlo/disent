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
import torch.nn.functional as F


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
    assert all(0 <= d < ndim for d in dims)
    # return new shape
    if ndim == 1:
        return [0], (1, *shape)
    # check resulting shape
    assert ndim >= 2
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
    moved_end_dims = tuple(dims_X[::-1])
    reshape_size = (size_B, size_X)
    # sort dims
    return moved_end_dims, reshape_size


def torch_dims_at_end_2d(tensor: torch.Tensor, dims: Union[int, Tuple[int, ...]] = -1, return_undo_data=True):
    # get dim info
    moved_end_dims, reshape_size = _get_2d_reshape_info(tensor.shape, dims=dims)
    # move all axes
    for d in moved_end_dims:
        tensor = torch.moveaxis(tensor, d, -1)
    moved_shape = tensor.shape
    # reshape
    tensor = torch.reshape(tensor, reshape_size)
    # return all info
    if return_undo_data:
        return tensor, moved_shape, moved_end_dims
    else:
        return tensor


def torch_undo_dims_at_end_2d(tensor: torch.Tensor, moved_shape, moved_end_dims):
    # reshape
    tensor = torch.reshape(tensor, moved_shape)
    # undo moving of dims
    for d in moved_end_dims[::-1]:
        tensor = torch.moveaxis(tensor, -1, d)
    # reshape
    return tensor


# ========================================================================= #
# Improved differentiable torchsort functions                               #
# ========================================================================= #


def torch_soft_sort(
    tensor: torch.Tensor,
    dims: Union[int, Tuple[int, ...]] = -1,
    regularization='l2',
    regularization_strength=1.0,
    dims_at_end=False,
):
    # we import it locally so that we don't have to install this
    import torchsort
    # reorder the dimensions
    tensor, moved_shape, moved_end_dims = torch_dims_at_end_2d(tensor, dims=dims, return_undo_data=True)
    # sort the last dimension of the 2D tensors
    tensor = torchsort.soft_sort(tensor, regularization=regularization, regularization_strength=regularization_strength)
    # undo the reorder operation
    if dims_at_end:
        return tensor
    return torch_undo_dims_at_end_2d(tensor, moved_shape=moved_shape, moved_end_dims=moved_end_dims)


def torch_soft_rank(
    tensor: torch.Tensor,
    dims: Union[int, Tuple[int, ...]] = -1,
    regularization='l2',
    regularization_strength=1.0,
    leave_dims_at_end=False,
):
    # we import it locally so that we don't have to install this
    import torchsort
    # reorder the dimensions
    tensor, moved_shape, moved_end_dims = torch_dims_at_end_2d(tensor, dims=dims, return_undo_data=True)
    # sort the last dimension of the 2D tensors
    tensor = torchsort.soft_rank(tensor, regularization=regularization, regularization_strength=regularization_strength)
    # undo the reorder operation
    if leave_dims_at_end:
        return tensor
    return torch_undo_dims_at_end_2d(tensor, moved_shape=moved_shape, moved_end_dims=moved_end_dims)


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
    pred = torch_dims_at_end_2d(pred, dims=dims, return_undo_data=False)
    targ = torch_dims_at_end_2d(targ, dims=dims, return_undo_data=False)
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
# loss functions                                                            #
# ========================================================================= #


def torch_mse_rank_loss(pred, targ, dims=-1, regularization='l2', regularization_strength=1.0, reduction='mean'):
    return F.mse_loss(
        torch_soft_rank(pred, dims=dims, regularization=regularization, regularization_strength=regularization_strength, leave_dims_at_end=False),
        torch_soft_rank(targ, dims=dims, regularization=regularization, regularization_strength=regularization_strength, leave_dims_at_end=False),
        reduction=reduction,
    )


# ========================================================================= #
# end                                                                       #
# ========================================================================= #
