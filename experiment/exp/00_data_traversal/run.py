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
from typing import Union

import numpy as np
from matplotlib import pyplot as plt

import experiment.exp.util as H
from disent.data.groundtruth import Cars3dData
from disent.data.groundtruth import DSpritesData
from disent.data.groundtruth import GroundTruthData
from disent.data.groundtruth import Shapes3dData
from disent.data.groundtruth import SmallNorbData
from disent.data.groundtruth import XYSquaresData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.util import TempNumpySeed


# ========================================================================= #
# core                                                                      #
# ========================================================================= #


def plot_dataset_traversals(
    gt_data: Union[GroundTruthData, GroundTruthDataset],
    f_idxs=None,
    num_cols=8,
    base_factors=None,
    add_random_traversal=True,
    pad=8,
    bg_color=127,
    border=False,
    rel_path=None,
    save=True,
    seed=777,
    plt_scale=4.5,
    offset=0.75
):
    if not isinstance(gt_data, GroundTruthDataset):
        gt_data = GroundTruthDataset(gt_data)
    f_idxs = H.get_factor_idxs(gt_data, f_idxs)
    # get traversal grid
    row_labels = [gt_data.factor_names[i] for i in f_idxs]
    grid = H.dataset_traversal_tasks(
        gt_data=gt_data,
        tasks='grid',
        data_mode='raw',
        factor_names=f_idxs,
        num=num_cols,
        seed=seed,
        base_factors=base_factors,
        traverse_mode='interval',
        pad=pad,
        bg_color=bg_color,
        border=border,
    )
    # add random traversal
    if add_random_traversal:
        with TempNumpySeed(seed):
            row_labels = ['random'] + row_labels
            grid = np.concatenate([
                gt_data.dataset_sample_batch(num_samples=num_cols, mode='raw')[None, ...],
                grid
            ])
    # add missing channel
    if grid.ndim == 4:
        grid = grid[..., None].repeat(3, axis=-1)
    assert grid.ndim == 5
    # make figure
    h, w, _, _, c = grid.shape
    assert c == 3
    fig, axs = H.plt_subplots_imshow(grid, row_labels=row_labels, subplot_padding=0.5, figsize=(offset + (1/2.54)*w*plt_scale, (1/2.54)*h*plt_scale))
    # save figure
    if save and (rel_path is not None):
        plt.savefig(H.make_rel_path_add_ext(rel_path, ext='.png'))
    plt.show()
    # done!
    return fig, axs


# ========================================================================= #
# entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))

    # options
    all_squares = True
    add_random_traversal = True
    num_cols = 7

    # save image
    for i in ([1, 2, 3, 4, 5, 6, 7, 8] if all_squares else [1, 8]):
        plot_dataset_traversals(
            XYSquaresData(grid_spacing=i, grid_size=8, no_warnings=True),
            rel_path=f'plots/xy-squares-traversal-spacing{i}',
            seed=7, add_random_traversal=add_random_traversal, num_cols=num_cols
        )

    plot_dataset_traversals(
        Shapes3dData(),
        rel_path=f'plots/shapes3d-traversal',
        seed=47, add_random_traversal=add_random_traversal, num_cols=num_cols
    )

    plot_dataset_traversals(
        DSpritesData(),
        rel_path=f'plots/dsprites-traversal',
        seed=47, add_random_traversal=add_random_traversal, num_cols=num_cols
    )

    plot_dataset_traversals(
        SmallNorbData(),
        rel_path=f'plots/smallnorb-traversal',
        seed=47, add_random_traversal=add_random_traversal, num_cols=num_cols
    )

    plot_dataset_traversals(
        Cars3dData(),
        rel_path=f'plots/cars3d-traversal',
        seed=47, add_random_traversal=add_random_traversal, num_cols=num_cols
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


