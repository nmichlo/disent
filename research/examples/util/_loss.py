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

import inspect
import warnings
from typing import Optional
from typing import Sequence

import numpy as np
import torch
from torch.nn import functional as F

from disent import registry
from disent.nn.loss.reduction import batch_loss_reduction




# ========================================================================= #
# Loss                                                                      #
# ========================================================================= #


def _unreduced_mse_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, targ, reduction='none')


def _unreduced_mae_loss(pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
    return torch.abs(pred - targ)


def unreduced_loss(pred: torch.Tensor, targ: torch.Tensor, mode='mse') -> torch.Tensor:
    return _LOSS_FNS[mode](pred, targ)


_LOSS_FNS = {
    'mse': _unreduced_mse_loss,
    'mae': _unreduced_mae_loss,
}


# ========================================================================= #
# Pairwise Loss                                                             #
# ========================================================================= #


def pairwise_loss(pred: torch.Tensor, targ: torch.Tensor, mode='mse', mean_dtype=None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # check input
    assert pred.shape == targ.shape
    # mean over final dims
    loss = unreduced_loss(pred=pred, targ=targ, mode=mode)
    # mask values
    if mask is not None:
        loss *= mask
    # reduce
    loss = batch_loss_reduction(loss, reduction_dtype=mean_dtype, reduction='mean')
    # check result
    assert loss.shape == pred.shape[:1]
    # done
    return loss


def unreduced_overlap(pred: torch.Tensor, targ: torch.Tensor, mode='mse') -> torch.Tensor:
    # -ve loss
    return - unreduced_loss(pred=pred, targ=targ, mode=mode)


def pairwise_overlap(pred: torch.Tensor, targ: torch.Tensor, mode='mse', mean_dtype=None) -> torch.Tensor:
    # -ve loss
    return - pairwise_loss(pred=pred, targ=targ, mode=mode, mean_dtype=mean_dtype)


# ========================================================================= #
# Factor Distances                                                          #
# ========================================================================= #


def np_factor_dists(
    factors_a: np.ndarray,
    factors_b: np.ndarray,
    factor_sizes: Optional[Sequence[int]] = None,
    circular_if_factor_sizes: bool = True,
    p: int = 1,
) -> np.ndarray:
    assert factors_a.ndim == 2
    assert factors_a.shape == factors_b.shape
    # compute factor distances
    fdists = np.abs(factors_a - factors_b)  # (NUM, FACTOR_SIZE)
    # circular distance
    if (factor_sizes is not None) and circular_if_factor_sizes:
        M = np.array(factor_sizes)[None, :]                       # (FACTOR_SIZE,) -> (1, FACTOR_SIZE)
        assert M.shape == (1, factors_a.shape[-1])
        fdists = np.where(fdists > (M // 2), M - fdists, fdists)  # (NUM, FACTOR_SIZE)
    # compute final dists
    fdists = (fdists ** p).sum(axis=-1) ** (1 / p)
    # return values
    return fdists  # (NUM,)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
