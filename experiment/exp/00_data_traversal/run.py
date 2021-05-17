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
import itertools
import os
from typing import Union

import imageio
import numpy as np
from matplotlib import pyplot as plt

from disent.data.groundtruth import Cars3dData
from disent.data.groundtruth import DSpritesData
from disent.data.groundtruth import GroundTruthData
from disent.data.groundtruth import Shapes3dData
from disent.data.groundtruth import SmallNorbData
from disent.data.groundtruth import XYSquaresData
from disent.dataset.groundtruth import GroundTruthDataset
from disent.util import TempNumpySeed
from disent.visualize.visualize_util import make_image_grid

from experiment.exp.util.io_util import make_rel_path_add_ext

# ========================================================================= #
# helper                                                                    #
# ========================================================================= #


def output_image(img, rel_path, save=True, plot=True):
    if save and (rel_path is not None):
        # convert image type
        if img.dtype in (np.float16, np.float32, np.float64, np.float128):
            assert np.all(img >= 0) and np.all(img <= 1.0)
            img = np.uint8(img * 255)
        elif img.dtype in (np.int16, np.int32, np.int64):
            assert np.all(img >= 0) and np.all(img <= 255)
            img = np.uint8(img)
        assert img.dtype == np.uint8, f'unsupported image dtype: {img.dtype}'
        # save image
        imageio.imsave(make_rel_path_add_ext(rel_path, ext='.png'), img)
    if plot:
        plt.imshow(img)
        plt.show()
    return img


def convert_f_idxs(gt_data, f_idxs):
    if f_idxs is None:
        f_idxs = list(range(gt_data.num_factors))
    else:
        f_idxs = [(gt_data.factor_names.index(i) if isinstance(i, str) else i) for i in f_idxs]
    return f_idxs


def make_traversal_grid(gt_data: Union[GroundTruthData, GroundTruthDataset], f_idxs=None, factors=True, num=8):
    # get defaults
    if not isinstance(gt_data, GroundTruthDataset):
        gt_data = GroundTruthDataset(gt_data)
    f_idxs = convert_f_idxs(gt_data, f_idxs)
    # sample factors
    if isinstance(factors, bool):
        factors = gt_data.sample_factors(1) if factors else None
    # sample traversals
    images = []
    for f_idx in f_idxs:
        fs = gt_data.sample_random_cycle_factors(f_idx, factors=factors, num=num)
        images.append(gt_data.dataset_batch_from_factors(fs, mode='raw') / 255.0)
    images = np.stack(images)
    # return grid
    return images # (F, N, H, W, C)


def make_dataset_traversals(
    gt_data,
    f_idxs=None, num_cols=8, factors=True,
    pad=8, bg_color=1.0, border=False,
    rel_path=None, save=True, plot=False,
    seed=777,
):
    with TempNumpySeed(seed):
        images = make_traversal_grid(gt_data, f_idxs=f_idxs, num=num_cols, factors=factors)
        image = make_image_grid(images.reshape(np.prod(images.shape[:2]), *images.shape[2:]), pad=pad, bg_color=bg_color, border=border, num_cols=num_cols)
        output_image(img=image, rel_path=rel_path, save=save, plot=plot)
    return image, images


# ========================================================================= #
# core                                                                      #
# ========================================================================= #


def plot_dataset_traversals(
    gt_data,
    f_idxs=None, num_cols=8, factors=True, add_random_traversal=True,
    pad=8, bg_color=1.0, border=False,
    rel_path=None, save=True, plot=True,
    seed=777,
    plt_scale=7, offset=0.75, plt_transpose=False,
):
    if not isinstance(gt_data, GroundTruthDataset):
        gt_data = GroundTruthDataset(gt_data)
    f_idxs = convert_f_idxs(gt_data, f_idxs)
    # print factors
    print(f'{gt_data.data.__class__.__name__}: loaded factors {tuple([gt_data.factor_names[i] for i in f_idxs])} of {gt_data.factor_names}')
    # get traversal grid
    _, images = make_dataset_traversals(
        gt_data,
        f_idxs=f_idxs, num_cols=num_cols, factors=factors,
        pad=pad, bg_color=bg_color, border=border,
        rel_path=None, save=False, plot=False,
        seed=seed,
    )
    # add random traversal
    if add_random_traversal:
        with TempNumpySeed(seed):
            ran_imgs = gt_data.dataset_sample_batch(num_samples=num_cols, mode='raw') / 255
            images = np.concatenate([ran_imgs[None, ...], images])
    # transpose
    if plt_transpose:
        images = np.transpose(images, [1, 0, *range(2, images.ndim)])
    # add missing channel
    if images.ndim == 4:
        images = images[..., None].repeat(3, axis=-1)
    assert images.ndim == 5
    # make figure
    oW, oH = (0, offset*0.5) if plt_transpose else (offset, 0)
    H, W, _, _, C = images.shape
    assert C == 3
    cm = 1 / 2.54
    fig, axs = plt.subplots(H, W, figsize=(oW + cm*W*plt_scale, oH + cm*H*plt_scale))
    axs = np.array(axs)
    # plot images
    for y, x in itertools.product(range(H), range(W)):
        img, ax = images[y, x], axs[y, x]
        ax.imshow(img)
        i, j = (y, x) if plt_transpose else (x, y)
        if (i == H-1) if plt_transpose else (i == 0):
            label = 'random' if (add_random_traversal and (j == 0)) else gt_data.factor_names[f_idxs[j-int(add_random_traversal)]]
            (ax.set_xlabel if plt_transpose else ax.set_ylabel)(label, fontsize=26)
        # ax.set_axis_off()
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
    plt.tight_layout()
    # save and show
    if save and (rel_path is not None):
        plt.savefig(make_rel_path_add_ext(rel_path, ext='.png'))
    if plot:
        plt.show()


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
            XYSquaresData(grid_spacing=i, max_placements=8, no_warnings=True),
            factors=None,
            rel_path=f'plots/xy-squares-traversal-spacing{i}',
            f_idxs=None, seed=7, add_random_traversal=add_random_traversal, num_cols=num_cols
        )

    plot_dataset_traversals(
        Shapes3dData(),
        factors=None,
        rel_path=f'plots/shapes3d-traversal',
        f_idxs=None, seed=47, add_random_traversal=add_random_traversal, num_cols=num_cols
    )

    plot_dataset_traversals(
        DSpritesData(),
        factors=None,
        rel_path=f'plots/dsprites-traversal',
        f_idxs=None, seed=47, add_random_traversal=add_random_traversal, num_cols=num_cols
    )

    plot_dataset_traversals(
        SmallNorbData(),
        factors=None,
        rel_path=f'plots/smallnorb-traversal',
        f_idxs=None, seed=47, add_random_traversal=add_random_traversal, num_cols=num_cols
    )

    plot_dataset_traversals(
        Cars3dData(),
        factors=None,
        rel_path=f'plots/cars3d-traversal',
        f_idxs=None, seed=47, add_random_traversal=add_random_traversal, num_cols=num_cols
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


