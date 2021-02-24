import numpy as np
import torch


# ========================================================================= #
# Reduction Strategies                                                      #
# ========================================================================= #


def loss_reduction_sum(x: torch.Tensor) -> torch.Tensor:
    return x.sum()


def loss_reduction_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean()


def loss_reduction_batch_mean(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1).sum(dim=-1).mean()


_LOSS_REDUCTION_STRATEGIES = {
    'none': lambda tensor: tensor,
    'sum': loss_reduction_sum,
    'mean': loss_reduction_mean,
    'batch_mean': loss_reduction_batch_mean,
}


def loss_reduction(tensor: torch.Tensor, reduction='batch_mean'):
    return _LOSS_REDUCTION_STRATEGIES[reduction](tensor)


# ========================================================================= #
# Reduction Strategies                                                      #
# ========================================================================= #


def get_mean_loss_scale(x: torch.Tensor, reduction: str):
    # check the dimensions if given
    assert 2 <= x.ndim <= 4, 'unsupported number of dims, must be one of: BxC, BxHxW, BxCxHxW'

    # get the loss scaling
    if reduction == 'batch_mean':
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

