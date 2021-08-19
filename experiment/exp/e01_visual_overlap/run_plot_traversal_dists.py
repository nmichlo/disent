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

from argparse import Namespace
from typing import Optional
from typing import Sequence
from typing import Union

import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

import experiment.exp.util as H
from disent.dataset.data import GroundTruthData


def _unique_pair_indices(max_idx: int, batch_size: int):
    # sample pairs
    idx_a, idx_b = torch.randint(0, max_idx, size=(2, batch_size))
    # remove similar
    different = (idx_a != idx_b)
    idx_a = idx_a[different]
    idx_b = idx_b[different]
    # return values
    return idx_a, idx_b


def sample_factor_traversal(
    gt_data: GroundTruthData,
    f_idx: Optional[int] = None,
    num_pairs_mul: int = 100,
):
    # load traversal
    t_factors = gt_data.sample_random_factor_traversal(f_idx=f_idx)
    t_idxs = gt_data.pos_to_idx(t_factors)
    t_obs = torch.stack([gt_data[i] for i in t_idxs])

    # check values
    idxs_a, idxs_b = _unique_pair_indices(len(t_idxs), len(t_idxs) * num_pairs_mul)
    deltas = F.mse_loss(t_obs[idxs_a], t_obs[idxs_b], reduction='none').mean(dim=[-3, -2, -1])
    fdists = torch.abs(torch.from_numpy(t_factors)[idxs_a] - torch.from_numpy(t_factors)[idxs_b]).sum(dim=-1)

    return Namespace(
        t_factors=t_factors,
        t_idxs=t_idxs,
        t_obs=t_obs,
        idxs_a=idxs_a,
        idxs_b=idxs_b,
        deltas=deltas,
        fdists=fdists,
    )


@torch.no_grad()
def plot(
    dataset_name: str = 'dsprites',
    num_traversal_sample: int = 100,
    f_idxs: Optional[Sequence[Union[int, str]]] = None,
    num_pairs_mul: int = 100,
):
    gt_data = H.make_data(dataset_name)

    if f_idxs is None:
        f_idxs = list(range(gt_data.num_factors))

    # make plot
    fig, axs = plt.subplots(2, len(f_idxs), figsize=(5 * len(f_idxs), 10))
    axs = np.array(axs).reshape((2, len(f_idxs)))
    fig.suptitle(f'{dataset_name}')

    # fill in plot
    for i, f_idx in enumerate(f_idxs):
        factor_name = gt_data.factor_names[f_idx]
        factor_size = gt_data.factor_sizes[f_idx]

        deltas, fdists = [], []
        for _ in tqdm(range(num_traversal_sample), desc=f'{dataset_name}: {factor_name}'):
            data = sample_factor_traversal(gt_data, f_idx=f_idx, num_pairs_mul=num_pairs_mul)
            deltas.append(data.deltas)
            fdists.append(data.fdists)
        deltas, fdists = torch.cat(deltas), torch.cat(fdists)

        # add data
        axs[0, i].set_title(f'{factor_name} ({factor_size})')
        axs[0, i].violinplot([deltas], vert=False)
        axs[1, i].scatter(deltas, fdists)

    # plot!
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))

    # plot xysquares with increasing overlap
    # plot('xysquares_8x8_s1', f_idxs=[1])
    # plot('xysquares_8x8_s2', f_idxs=[1])
    # plot('xysquares_8x8_s3', f_idxs=[1])
    # plot('xysquares_8x8_s4', f_idxs=[1])
    # plot('xysquares_8x8_s5', f_idxs=[1])
    # plot('xysquares_8x8_s6', f_idxs=[1])
    # plot('xysquares_8x8_s7', f_idxs=[1])
    # plot('xysquares_8x8_s8', f_idxs=[1])

    # plot other datasets
    plot('dsprites')
    plot('cars3d')
    plot('shapes3d')
    plot('smallnorb')

    plot('xyblocks')
    plot('xyobject')
