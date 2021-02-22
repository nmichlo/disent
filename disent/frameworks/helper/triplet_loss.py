from dataclasses import dataclass
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


def elem_triplet_loss(anc, pos, neg, margin_min=None, margin_max=1., p=1):
    """
    Element-Wise Triplet Loss
    - THIS IS NOT EXPECTED TO WORK AT ALL!
    """
    return dist_elem_triplet_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


def dist_elem_triplet_loss(pos_delta, neg_delta, margin_min=None, margin_max=1., p=1):
    """
    Element-Wise Triplet Loss
    - THIS IS NOT EXPECTED TO WORK AT ALL!
    """
    if margin_min is not None:
        warnings.warn('elem_triplet_loss does not support margin_min')
    if p != 1:
        warnings.warn('elem_triplet_loss only supported p=1')
    p_dist = torch.abs(pos_delta)
    n_dist = torch.abs(neg_delta)
    loss = torch.clamp_min(p_dist - n_dist + margin_max, 0)
    return loss.mean()


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


def min_margin_triplet_loss(anc, pos, neg, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    """
    return dist_min_margin_triplet_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


def dist_min_margin_triplet_loss(pos_delta, neg_delta, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    """
    p_dist = torch.norm(pos_delta + margin_min, p=p, dim=-1)
    n_dist = torch.norm(neg_delta, p=p, dim=-1)
    loss = torch.clamp_min(p_dist - n_dist + margin_max, 0)
    return loss.mean()


# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


def min_clamp_triplet_loss(anc, pos, neg, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    """
    return dist_min_clamp_triplet_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


def dist_min_clamp_triplet_loss(pos_delta, neg_delta, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    """
    p_dist = torch.norm(pos_delta, p=p, dim=-1)
    n_dist = torch.norm(neg_delta, p=p, dim=-1)
    loss = torch.clamp_min(torch.clamp_min(p_dist, margin_min) - n_dist + margin_max, 0)
    return loss.mean()

# -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


def clamped_triplet_loss(anc, pos, neg, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    """
    return dist_clamped_triplet_loss(anc - pos, anc - neg, margin_min=margin_min, margin_max=margin_max, p=p)


def dist_clamped_triplet_loss(pos_delta, neg_delta, margin_min=0.01, margin_max=1., p=1):
    """
    Min Margin Triplet Loss
    """
    p_dist = torch.norm(pos_delta, p=p, dim=-1)
    n_dist = torch.norm(neg_delta, p=p, dim=-1)
    loss = torch.clamp_min(p_dist, margin_min) - torch.clamp_max(n_dist, margin_max)
    return loss.mean()


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
    'elem_triplet': elem_triplet_loss,
    'min_margin_triplet': min_margin_triplet_loss,
    'min_clamp_triplet': min_clamp_triplet_loss,
    'clamped_triplet': clamped_triplet_loss,
}


_DIST_TRIPLET_LOSSES = {
    'triplet': dist_triplet_loss,
    'elem_triplet': dist_elem_triplet_loss,
    'min_margin_triplet': dist_min_margin_triplet_loss,
    'min_clamp_triplet': dist_min_clamp_triplet_loss,
    'clamped_triplet': dist_clamped_triplet_loss,
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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #









