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
# Reduction Strategies                                                      #
# ========================================================================= #


def loss_reduction_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum()


def loss_reduction_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean()


def loss_reduction_mean_sum(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).sum(dim=-1).mean()


_LOSS_REDUCTION_STRATEGIES = {
    # 'none': lambda tensor: tensor,
    'sum': loss_reduction_sum,
    'mean': loss_reduction_mean,
    'mean_sum': loss_reduction_mean_sum,
}


def loss_reduction(tensor: torch.Tensor, reduction='mean'):
    return _LOSS_REDUCTION_STRATEGIES[reduction](tensor)


# ========================================================================= #
# Reduction Strategies                                                      #
# ========================================================================= #


def get_mean_loss_scale(x: torch.Tensor, reduction: str):
    # check the dimensions if given
    assert 2 <= x.ndim <= 4, 'unsupported number of dims, must be one of: BxC, BxHxW, BxCxHxW'

    # get the loss scaling
    if reduction == 'mean_sum':
        return np.prod(x.shape[1:])  # MEAN(B, SUM(C x H x W))
    elif reduction == 'mean':
        return 1
    elif reduction == 'sum':
        return np.prod(x.shape)  # SUM(B x C x H x W)
    else:
        raise KeyError('unsupported loss reduction mode')


# ========================================================================= #
# END                                                                 #
# ========================================================================= #

