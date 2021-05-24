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

from dataclasses import dataclass
from typing import Sequence
from typing import Type, Union
import torch
import warnings


# ========================================================================= #
# Triplet Modifications                                                     #
# ========================================================================= #


def triplet_loss(anc, pos, neg, margin_min=None, margin_max=1., p=1):
    """
    Standard Triplet Loss
    """
    return dist_triplet_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


def dist_triplet_loss(pos_delta, neg_delta, margin_min=None, margin_max=1., p=1):
    """
    Standard Triplet Loss
    """
    if margin_min is not None:
        warnings.warn('triplet_loss does not support margin_min')
    p_dist = torch.norm(pos_delta, p=p, dim=-1)
    n_dist = torch.norm(neg_delta, p=p, dim=-1)
    loss = torch.clamp_min(p_dist - n_dist + margin_max, 0)
    return loss.mean()


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


def triplet_sigmoid_loss(anc, pos, neg, margin_min=None, margin_max=1., p=1):
    """
    Sigmoid Triplet Loss
    https://arxiv.org/pdf/2003.14021.pdf
    """
    return dist_triplet_sigmoid_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


def dist_triplet_sigmoid_loss(pos_delta, neg_delta, margin_min=None, margin_max=1., p=1):
    """
    Sigmoid Triplet Loss
    https://arxiv.org/pdf/2003.14021.pdf
    """
    if margin_min is not None:
        warnings.warn('triplet_loss does not support margin_min')
    p_dist = torch.norm(pos_delta, p=p, dim=-1)
    n_dist = torch.norm(neg_delta, p=p, dim=-1)
    loss = torch.sigmoid((1/margin_max) * (p_dist - n_dist))
    return loss.mean()


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


# def elem_triplet_loss(anc, pos, neg, margin_min=None, margin_max=1., p=1):
#     """
#     Element-Wise Triplet Loss
#     TODO: THIS SHOULD NOT WORK AT ALL! JUST TRYING SOMETHING
#     """
#     return dist_elem_triplet_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


# def dist_elem_triplet_loss(pos_delta, neg_delta, margin_min=None, margin_max=1., p=1):
#     """
#     Element-Wise Triplet Loss
#     TODO: THIS SHOULD NOT WORK AT ALL! JUST TRYING SOMETHING
#     """
#     if margin_min is not None:
#         warnings.warn('elem_triplet_loss does not support margin_min')
#     if p != 1:
#         warnings.warn('elem_triplet_loss only supported p=1')
#     p_dist = torch.abs(pos_delta)
#     n_dist = torch.abs(neg_delta)
#     loss = torch.clamp_min(p_dist - n_dist + margin_max, 0)
#     return loss.mean()


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


# def min_margin_triplet_loss(anc, pos, neg, margin_min=0.01, margin_max=1., p=1):
#     """
#     Min Margin Triplet Loss
#     # TODO: this is wrong?
#     """
#     return dist_min_margin_triplet_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


# def dist_min_margin_triplet_loss(pos_delta, neg_delta, margin_min=0.01, margin_max=1., p=1):
#     """
#     Min Margin Triplet Loss
#     # TODO: this is wrong?
#     """
#     p_dist = torch.norm(pos_delta + margin_min, p=p, dim=-1)
#     n_dist = torch.norm(neg_delta, p=p, dim=-1)
#     loss = torch.clamp_min(p_dist - n_dist + margin_max, 0)
#     return loss.mean()


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


def min_clamped_triplet_loss(anc, pos, neg, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    TODO: is this better, or clamped_triplet_loss?
    """
    return dist_min_clamped_triplet_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


def dist_min_clamped_triplet_loss(pos_delta, neg_delta, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    TODO: is this better, or dist_clamped_triplet_loss?
    """
    p_dist = torch.norm(pos_delta, p=p, dim=-1)
    n_dist = torch.norm(neg_delta, p=p, dim=-1)
    loss = torch.clamp_min(torch.clamp_min(p_dist, margin_min) - n_dist + margin_max, 0)
    return loss.mean()

# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


def split_clamped_triplet_loss(anc, pos, neg, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    TODO: is this better, or min_clamp_triplet_loss?
    """
    return dist_split_clamped_triplet_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


def dist_split_clamped_triplet_loss(pos_delta, neg_delta, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    TODO: is this better, or dist_min_clamp_triplet_loss?
    """
    p_dist = torch.norm(pos_delta, p=p, dim=-1)
    n_dist = torch.norm(neg_delta, p=p, dim=-1)
    loss = torch.clamp_min(p_dist, margin_min) - torch.clamp_max(n_dist, margin_max) + (margin_max - margin_min)
    return loss.mean()


# ========================================================================= #
# EXTRA TRIPLET LOSSES                                                      #
# ========================================================================= #


# def triplet_lossless(anc, pos, neg, margin_min=None, margin_max=3., p=1, epsilon=1e-8):
#     """
#     TODO: inputs should be from sigmoid output layer.
#     https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
#     """
#     return dist_triplet_lossless(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p, epsilon=epsilon)


# def dist_triplet_lossless(pos_delta, neg_delta, margin_min=None, margin_max=3., p=1, epsilon=1e-8):
#     """
#     TODO: inputs should be from sigmoid output layer.
#     https://towardsdatascience.com/lossless-triplet-loss-7e932f990b24
#     """
#     if margin_min is not None:
#         warnings.warn('triplet_loss does not support margin_min')
#     p_dist = torch.norm(pos_delta, p=p, dim=-1)
#     n_dist = torch.norm(neg_delta, p=p, dim=-1)
#     # rename values
#     N = margin_max
#     # recommended value
#     beta = N
#     # non-linear activation
#     p_dist = - torch.log(-((    p_dist) / beta) + (1 + epsilon))
#     n_dist = - torch.log(-((N - n_dist) / beta) + (1 + epsilon))
#     # compute mean
#     return (p_dist + n_dist).mean()


# ========================================================================= #
# Get Triplet                                                               #
# ========================================================================= #


@dataclass
class TripletLossConfig(object):
    triplet_loss: str = 'triplet'
    triplet_margin_min: float = 0.1
    triplet_margin_max: float = 10
    triplet_scale: float = 100
    triplet_p: int = 2


_TRIPLET_LOSSES = {
    'triplet': triplet_loss,
    'triplet_sigmoid': triplet_sigmoid_loss,
    # 'elem_triplet': elem_triplet_loss,
    # 'min_margin_triplet': min_margin_triplet_loss,
    'min_clamped_triplet': min_clamped_triplet_loss,
    'split_clamped_triplet': split_clamped_triplet_loss,
    # 'triplet_lossless': triplet_lossless,
}


_DIST_TRIPLET_LOSSES = {
    'triplet': dist_triplet_loss,
    'triplet_sigmoid': dist_triplet_sigmoid_loss,
    # 'elem_triplet': dist_elem_triplet_loss,
    # 'min_margin_triplet': dist_min_margin_triplet_loss,
    'min_clamped_triplet': dist_min_clamped_triplet_loss,
    'split_clamped_triplet': dist_split_clamped_triplet_loss,
    # 'triplet_lossless': dist_triplet_lossless,
}


TripletConfigTypeHint = Union[TripletLossConfig, Type[TripletLossConfig]]


def configured_triplet(anc, pos, neg, cfg: TripletConfigTypeHint):
    return _TRIPLET_LOSSES[cfg.triplet_loss](
        anc, pos, neg,
        margin_min=cfg.triplet_margin_min,
        margin_max=cfg.triplet_margin_max,
        p=cfg.triplet_p,
    ) * cfg.triplet_scale


def configured_dist_triplet(pos_delta, neg_delta, cfg: TripletConfigTypeHint):
    return _DIST_TRIPLET_LOSSES[cfg.triplet_loss](
        pos_delta, neg_delta,
        margin_min=cfg.triplet_margin_min,
        margin_max=cfg.triplet_margin_max,
        p=cfg.triplet_p,
    ) * cfg.triplet_scale


def compute_triplet_loss(zs: Sequence[torch.Tensor], cfg: TripletConfigTypeHint):
    anc, pos, neg = zs
    # loss is scaled and everything
    loss = configured_triplet(anc, pos, neg, cfg=cfg)
    # return loss & log
    return loss, {
        f'{cfg.triplet_loss}_L{cfg.triplet_p}': loss
    }


def compute_dist_triplet_loss(zs_deltas: Sequence[torch.Tensor], cfg: TripletConfigTypeHint):
    pos_delta, neg_delta = zs_deltas
    # loss is scaled and everything
    loss = configured_dist_triplet(pos_delta, neg_delta, cfg=cfg)
    # return loss & log
    return loss, {
        f'{cfg.triplet_loss}_L{cfg.triplet_p}': loss
    }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #








