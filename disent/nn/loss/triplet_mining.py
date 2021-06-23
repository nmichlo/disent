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

import logging
from typing import Callable
from typing import Protocol
from typing import Tuple

import numpy as np
import torch


log = logging.getLogger(__name__)


# ========================================================================= #
# Mining Modes                                                              #
# ========================================================================= #


def _delta_mine_none(dist_ap: torch.Tensor, dist_an: torch.Tensor, top_k: int, margin_max: float):
    assert len(dist_ap) == len(dist_an)
    return torch.arange(len(dist_ap))


def _delta_mine_semi_hard_neg(dist_ap: torch.Tensor, dist_an: torch.Tensor, top_k: int, margin_max: float):
    # SEMI HARD NEGATIVE MINING
    # "choose an anchor-negative pair that is farther than the anchor-positive pair, but within the margin, and so still contributes a positive loss"
    # -- triples satisfy d(a, p) < d(a, n) < alpha
    semi_hard_mask = (dist_ap < dist_an) & (dist_an < margin_max)
    semi_hard_idxs = torch.arange(len(semi_hard_mask))[semi_hard_mask]
    return semi_hard_idxs


def _delta_mine_hard_neg(dist_ap: torch.Tensor, dist_an: torch.Tensor, top_k: int, margin_max: float):
    # HARD NEGATIVE MINING
    # "most similar images which have a different label from the anchor image"
    # -- triples with smallest d(a, n)
    hard_idxs = torch.argsort(dist_an, descending=False)[:int(top_k)]
    return hard_idxs


def _delta_easy_neg(dist_ap, dist_an, cfg):
    # EASY NEGATIVE MINING
    # "least similar images which have the different label from the anchor image"
    raise RuntimeError('This triplet mode is not useful! Choose another.')


def _delta_mine_hard_pos(dist_ap: torch.Tensor, dist_an: torch.Tensor, top_k: int, margin_max: float):
    # HARD POSITIVE MINING -- this performs really well!
    # "least similar images which have the same label to as anchor image"
    # -- shown not to be suitable for all datasets
    hard_idxs = torch.argsort(dist_ap, descending=True)[:int(top_k)]
    return hard_idxs


def _delta_mine_easy_pos(dist_ap: torch.Tensor, dist_an: torch.Tensor, top_k: int, margin_max: float):
    # EASY POSITIVE MINING
    # "the most similar images that have the same label as the anchor image"
    easy_idxs = torch.argsort(dist_ap, descending=False)[:int(top_k)]
    return easy_idxs


_TRIPLET_MINE_MODES = {
    'none':          _delta_mine_none,
    'semi_hard_neg': _delta_mine_semi_hard_neg,
    'hard_neg':      _delta_mine_hard_neg,
    # 'easy_neg':    delta_mine_easy_neg,  # not actually useful
    'hard_pos':      _delta_mine_hard_pos,
    'easy_pos':      _delta_mine_easy_pos,
}


# ========================================================================= #
# General Miner                                                             #
# ========================================================================= #


@torch.no_grad()
def mine(mode: str, dist_ap: torch.Tensor, dist_an: torch.Tensor, top_k: int, margin_max: float) -> torch.Tensor:
    # check arrays
    assert (dist_ap.ndim == 1) and (dist_an.ndim == 1), f'dist arrays must only have one dimension: dist_ap: {dist_ap.shape} & dist_an: {dist_an.shape}'
    assert (dist_ap.shape == dist_an.shape), f'dist array shapes do not match: {dist_ap.shape} & dist_an: {dist_an.shape}'
    # get mining function
    try:
        mine_fn = _TRIPLET_MINE_MODES[mode]
    except KeyError:
        raise KeyError(f'invalid triplet mining mode: {repr(mode)}, must be one of: {sorted(_TRIPLET_MINE_MODES.keys())}')
    # mine indices
    idxs = mine_fn(dist_ap=dist_ap, dist_an=dist_an, top_k=top_k, margin_max=margin_max)
    # check and return values
    if len(idxs) > 0:
        return idxs
    else:
        log.warning('no results using {repr(mode)} mining! using entire batch instead')
        return _delta_mine_none(dist_ap=dist_ap, dist_an=dist_an, top_k=top_k, margin_max=margin_max)


def mine_random_mode(mode: str, dist_ap: torch.Tensor, dist_an: torch.Tensor, top_k: int, margin_max: float):
    # randomly choose a mode
    # eg. `ran:hard_pos+easy_pos` randomly chooses between `hard_pos` and `easy_pos`
    if mode.startswith('ran:'):
        mode = np.random.choice(mode[len('ran:'):].split('+'))
    # mine like usual
    return mine(mode=mode, dist_ap=dist_ap, dist_an=dist_an, top_k=top_k, margin_max=margin_max)


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


class SampledTripletMineCfgProto(Protocol):
    triplet_margin_max: float
    overlap_num: int
    overlap_mine_triplet_mode: str
    overlap_mine_ratio: float


def configured_mine(dist_ap: torch.Tensor, dist_an: torch.Tensor, cfg: SampledTripletMineCfgProto) -> torch.Tensor:
    return mine_random_mode(
        mode=cfg.overlap_mine_triplet_mode,
        dist_ap=dist_ap,
        dist_an=dist_an,
        top_k=int(cfg.overlap_num * cfg.overlap_mine_ratio),
        margin_max=cfg.triplet_margin_max
    )


@torch.no_grad()
def configured_idx_mine(
    x_targ: torch.Tensor,
    a_idxs: torch.Tensor,
    p_idxs: torch.Tensor,
    n_idxs: torch.Tensor,
    cfg: SampledTripletMineCfgProto,
    pairwise_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],  # should return arrays with ndim == 1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO: this function is quite useless, its easier to just use configured_mine_random_mode
    # compute differences
    dist_ap = pairwise_loss_fn(x_targ[a_idxs], x_targ[p_idxs])
    dist_an = pairwise_loss_fn(x_targ[a_idxs], x_targ[n_idxs])
    # mine indices
    idxs = configured_mine(dist_ap=dist_ap, dist_an=dist_an, cfg=cfg)
    # check & return values
    return a_idxs[idxs], p_idxs[idxs], n_idxs[idxs]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
