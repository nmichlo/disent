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
from argparse import Namespace
from collections import defaultdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import experiment.exp.util as H
from disent.dataset.data import GroundTruthData


# ========================================================================= #
# Factor Traversal Stats                                                    #
# ========================================================================= #


SampleModeHint = Union[Literal['random'], Literal['near'], Literal['combinations']]


@torch.no_grad()
def sample_factor_traversal_info(
    gt_data: GroundTruthData,
    f_idx: Optional[int] = None,
    circular_distance: bool = False,
    sample_mode: SampleModeHint = 'random',
) -> dict:
    # load traversal -- TODO: this is the bottleneck! not threaded
    factors, indices, obs = gt_data.sample_random_obs_traversal(f_idx=f_idx, obs_collect_fn=torch.stack)
    # get pairs
    idxs_a, idxs_b = H.pair_indices(max_idx=len(indices), mode=sample_mode)
    # compute deltas
    deltas = F.mse_loss(obs[idxs_a], obs[idxs_b], reduction='none').mean(dim=[-3, -2, -1]).numpy()
    fdists = H.np_factor_dists(factors[idxs_a], factors[idxs_b], factor_sizes=gt_data.factor_sizes, circular_if_factor_sizes=circular_distance, p=1)
    # done!
    return dict(
        # traversals
        factors=factors,    # np.ndarray
        indices=indices,    # np.ndarray
        obs=obs,            # torch.Tensor
        # pairs
        idxs_a=idxs_a,      # np.ndarray
        idxs_b=idxs_b,      # np.ndarray
        deltas=deltas,      # np.ndarray
        fdists=fdists,      # np.ndarray
    )


def sample_factor_traversal_info_and_distmat(
    gt_data: GroundTruthData,
    f_idx: Optional[int] = None,
    circular_distance: bool = False,
) -> dict:
    dat = sample_factor_traversal_info(gt_data=gt_data, f_idx=f_idx, sample_mode='combinations', circular_distance=circular_distance)
    # extract
    factors, idxs_a, idxs_b, deltas, fdists = dat['factors'], dat['idxs_a'], dat['idxs_b'], dat['deltas'], dat['fdists']
    # generate deltas matrix
    deltas_matrix = np.zeros([factors.shape[0], factors.shape[0]])
    deltas_matrix[idxs_a, idxs_b] = deltas
    deltas_matrix[idxs_b, idxs_a] = deltas
    # generate distance matrix
    fdists_matrix = np.zeros([factors.shape[0], factors.shape[0]])
    fdists_matrix[idxs_a, idxs_b] = fdists
    fdists_matrix[idxs_b, idxs_a] = fdists
    # done!
    return dict(**dat, deltas_matrix=deltas_matrix, fdists_matrix=fdists_matrix)


# ========================================================================= #
# Factor Traversal Collector                                                #
# ========================================================================= #


def _collect_stats_for_factors(
    gt_data: GroundTruthData,
    f_idxs: Sequence[int],
    stats_fn: Callable[[GroundTruthData, int], Dict[str, Any]],
    keep_keys: Sequence[str],
    stats_callback: Optional[Callable[[Dict[str, List[Any]], int], None]] = None,
    return_stats: bool = True,
    num_traversal_sample: int = 100,
) -> List[Dict[str, List[Any]]]:
    # prepare
    f_idxs = H.normalise_factor_idxs(gt_data, factors=f_idxs)
    # generate data per factor
    f_stats = []
    for i, f_idx in enumerate(f_idxs):
        factor_name = gt_data.factor_names[f_idx]
        factor_size = gt_data.factor_sizes[f_idx]
        # repeatedly generate stats per factor
        stats = defaultdict(list)
        for _ in tqdm(range(num_traversal_sample), desc=f'{gt_data.name}: {factor_name}'):
            data = stats_fn(gt_data, f_idx)
            for key in keep_keys:
                stats[key].append(data[key])
        # save factor stats
        if return_stats:
            f_stats.append(stats)
        if stats_callback:
            stats_callback(stats, f_idx)
    # done!
    if return_stats:
        return f_stats


# ========================================================================= #
# Plot Traversal Stats                                                      #
# ========================================================================= #


def plot_traversal_stats(
    dataset_name: str,
    num_traversal_sample: int = 256,
    f_idxs: Optional[H.NonNormalisedFactors] = None,
    circular_distance: bool = False,
):
    def stats_fn(gt_data, f_idx):
        return sample_factor_traversal_info_and_distmat(gt_data=gt_data, f_idx=f_idx, circular_distance=circular_distance)

    def plot_ax(stats: dict, f_idx: int):
        deltas = np.concatenate(stats['deltas'])
        fdists = np.concatenate(stats['fdists'])
        fdists_matrix = np.mean(stats['fdists_matrix'], axis=0)
        deltas_matrix = np.mean(stats['deltas_matrix'], axis=0)
        # subplot!
        ax0, ax1, ax2, ax3 = axs[:, f_idx]
        ax0.set_title(f'{gt_data.factor_names[f_idx]} ({gt_data.factor_sizes[f_idx]})')

        ax0.set_title('fdists proportions')
        ax0.violinplot([deltas], vert=False)
        ax0.set_xlabel('deltas')
        ax0.set_ylabel('proportion')

        ax1.set_title('deltas vs. fdists')
        ax1.scatter(x=deltas, y=fdists, s=3)
        ax1.set_xlabel('deltas')
        ax1.set_ylabel('fdists')

        ax2.set_title('fdists')
        ax2.imshow(fdists_matrix)
        ax2.set_xlabel('f_idx')
        ax2.set_ylabel('f_idx')

        ax3.set_title('divergence')
        ax3.imshow(deltas_matrix)
        ax3.set_xlabel('f_idx')
        ax3.set_ylabel('f_idx')

    # prepare
    gt_data = H.make_data(dataset_name)
    f_idxs = H.normalise_factor_idxs(gt_data, factors=f_idxs)
    # settingss
    r, c = [4,  len(f_idxs)]
    h, w = [16, len(f_idxs)*4]
    # initialize plot
    fig, axs = plt.subplots(r, c, figsize=(w, h), squeeze=False)
    fig.suptitle(f'{dataset_name} [circular={circular_distance}]')
    # generate plot
    _collect_stats_for_factors(
        gt_data=gt_data,
        f_idxs=f_idxs,
        stats_fn=stats_fn,
        keep_keys=['deltas', 'fdists', 'deltas_matrix', 'fdists_matrix'],
        stats_callback=plot_ax,
        num_traversal_sample=num_traversal_sample,
    )
    # finalize plot
    fig.tight_layout()
    plt.show()


# ========================================================================= #
# ENTRY                                                                     #
# ========================================================================= #


if __name__ == '__main__':
    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))

    # plot xysquares with increasing overlap
    # plot_traversal_stats('xysquares_8x8_s1', f_idxs=[1])
    # plot_traversal_stats('xysquares_8x8_s2', f_idxs=[1])
    # plot_traversal_stats('xysquares_8x8_s3', f_idxs=[1])
    # plot_traversal_stats('xysquares_8x8_s4', f_idxs=[1])
    # plot_traversal_stats('xysquares_8x8_s5', f_idxs=[1])
    # plot_traversal_stats('xysquares_8x8_s6', f_idxs=[1])
    # plot_traversal_stats('xysquares_8x8_s7', f_idxs=[1])
    # plot_traversal_stats('xysquares_8x8_s8', f_idxs=[1])

    # plot other datasets
    plot_traversal_stats('dsprites')
    plot_traversal_stats('cars3d')
    plot_traversal_stats('shapes3d')
    plot_traversal_stats('smallnorb')

    # plot_traversal_stats('xyblocks')
    # plot_traversal_stats('xyobject')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
