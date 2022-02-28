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


import os
from collections import defaultdict
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import seaborn as sns
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, pearsonr

from disent.nn.loss.reduction import batch_loss_reduction
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import torch.nn.functional as F

import research.code.util as H
from disent.dataset import DisentDataset
from disent.dataset.data import Cars3d64Data
from disent.dataset.data import DSpritesData
from disent.dataset.data import GroundTruthData
from disent.dataset.data import Shapes3dData
from disent.dataset.data import SmallNorb64Data
from research.code.dataset.data import XYSquaresData
from disent.dataset.transform import ToImgTensorF32
from disent.util import to_numpy
from disent.util.function import wrapped_partial


# ========================================================================= #
# plot                                                                      #
# ========================================================================= #



# NOTE: this should match _factored_components._dists_compute_scores!
#       -- this is taken directly from there!
def _compute_rcorr_ground_data(num_triplets: int, xs_traversal: torch.Tensor, factors: torch.Tensor) -> Tuple[float, float]:
    # checks
    assert len(factors) == len(xs_traversal)
    assert factors.device == xs_traversal.device
    # generate random triplets
    # - {p, n} indices do not need to be sorted like triplets, these can be random.
    #   This metric is symmetric for swapped p & n values.
    idxs_a, idxs_p, idxs_n = torch.randint(0, len(xs_traversal), size=(3, num_triplets), device=xs_traversal.device)
    # compute distances -- shape: (num,)
    ap_ground_dists = torch.norm(factors[idxs_a, :] - factors[idxs_p, :], p=1, dim=-1)
    an_ground_dists = torch.norm(factors[idxs_a, :] - factors[idxs_n, :], p=1, dim=-1)
    ap_data_dists = batch_loss_reduction(F.mse_loss(xs_traversal[idxs_a, ...], xs_traversal[idxs_p, ...], reduction='none'), reduction_dtype=torch.float32, reduction='mean')
    an_data_dists = batch_loss_reduction(F.mse_loss(xs_traversal[idxs_a, ...], xs_traversal[idxs_n, ...], reduction='none'), reduction_dtype=torch.float32, reduction='mean')
    # concatenate values -- shape: (2 * num,)
    ground_dists    = torch.cat([ap_ground_dists,    an_ground_dists],    dim=0).numpy()
    data_dists      = torch.cat([ap_data_dists,      an_data_dists],      dim=0).numpy()
    # compute rcorr scores -- shape: ()
    # - compute the pearson rank correlation coefficient over the concatenated distances
    linear_corr, _ = pearsonr(ground_dists, data_dists)
    rank_corr, _ = spearmanr(ground_dists, data_dists)
    # return values -- shape: ()
    return linear_corr, rank_corr


def _compute_mean_rcorr_ground_data(dataset: DisentDataset, f_idx: Optional[Union[str, int]], num_triplets: int, repeats: int, progress: bool = True, random_batch_size: Optional[int] = 64):
    # normalise everything!
    if f_idx in ('random', None):
        f_idx = None
        f_name = 'random'
    elif isinstance(f_idx, str):
        f_idx = dataset.gt_data.normalise_factor_idx(f_idx)
        f_name = dataset.gt_data.factor_names[f_idx]
    else:
        assert isinstance(f_idx, int)
        f_name = dataset.gt_data.factor_names[f_idx]
    # get defaults
    if random_batch_size is None:
        random_batch_size = int(np.mean(dataset.gt_data.factor_sizes))
    # compute averages
    correlations_linear, correlations_rank = [], []
    for i in tqdm(range(repeats), desc=f'{dataset.gt_data.name}: {f_name}', disable=not progress):
        # sample random factors
        if f_idx is None:
            factors = dataset.gt_data.sample_factors(size=random_batch_size)
        else:
            factors = dataset.gt_data.sample_random_factor_traversal(f_idx=f_idx)
        # encode factors
        xs = dataset.dataset_batch_from_factors(factors, 'input').cpu()
        factors = torch.from_numpy(factors).to(torch.float32).cpu()
        # [COMPUTE SAME RATIO & CORRELATION]
        linear_corr, rank_corr = _compute_rcorr_ground_data(num_triplets, xs_traversal=xs, factors=factors)
        # [UPDATE SCORES]
        correlations_linear.append(linear_corr)
        correlations_rank.append(rank_corr)
    # return the average!
    return np.mean(correlations_linear), np.mean(correlations_rank)


# ========================================================================= #
# entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    gt_data_classes = {
      # 'XYObject':  wrapped_partial(XYObjectData),
      # 'XYBlocks':  wrapped_partial(XYBlocksData),
      #   'XYSquares': wrapped_partial(XYSquaresData),
        'DSprites':  wrapped_partial(DSpritesData),
        'Shapes3d':  wrapped_partial(Shapes3dData),
        'Cars3d':    wrapped_partial(Cars3d64Data),
        'SmallNorb': wrapped_partial(SmallNorb64Data),
      # 'Mpi3d':     wrapped_partial(Mpi3dData),

        'XYSquares-1-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=1, grid_size=8, no_warnings=True),
        'XYSquares-2-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=2, grid_size=8, no_warnings=True),
        'XYSquares-3-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=3, grid_size=8, no_warnings=True),
        'XYSquares-4-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=4, grid_size=8, no_warnings=True),
        'XYSquares-5-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=5, grid_size=8, no_warnings=True),
        'XYSquares-6-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=6, grid_size=8, no_warnings=True),
        'XYSquares-7-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=7, grid_size=8, no_warnings=True),
        'XYSquares-8-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=8, grid_size=8, no_warnings=True),
    }

    num_triplets = 256
    repeats = 2048
    progress = False

    for name, data_cls in  gt_data_classes.items():
        dataset = DisentDataset(data_cls(), transform=ToImgTensorF32(size=64))
        factor_names = (*dataset.gt_data.factor_names, 'random')
        # compute over each factor name
        for f_name in [*dataset.gt_data.factor_names, 'random']:
            linear_corr, rank_corr = _compute_mean_rcorr_ground_data(dataset, f_idx=f_name, num_triplets=num_triplets, repeats=repeats, progress=progress)
            print(f'[{name}] f_idx={f_name:{max(len(s) for s in factor_names)}s} linear_corr={linear_corr:7.5f}, rank_corr={rank_corr:7.5f}')
        print()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
