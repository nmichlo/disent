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

from numbers import Number
from typing import Any
from typing import Dict
from typing import Optional

import numpy as np
import torch
import logging

from disent.dataset import DisentDataset
from disent.dataset.util.state_space import NonNormalisedFactors
from disent.util.seeds import TempNumpySeed
from disent.util.visualize.vis_util import make_animated_image_grid
from disent.util.visualize.vis_util import make_image_grid

# TODO: matplotlib is not in requirements
from matplotlib import pyplot as plt


log = logging.getLogger(__name__)


# ========================================================================= #
# images                                                                    #
# ========================================================================= #


# TODO: similar functions exist: output_image, to_img, to_imgs, reconstructions_to_images
def to_img(x: torch.Tensor, scale=False, to_cpu=True, move_channels=True) -> torch.Tensor:
    assert x.ndim == 3, 'image must have 3 dimensions: (C, H, W)'
    return to_imgs(x, scale=scale, to_cpu=to_cpu, move_channels=move_channels)


# TODO: similar functions exist: output_image, to_img, to_imgs, reconstructions_to_images
def to_imgs(x: torch.Tensor, scale=False, to_cpu=True, move_channels=True) -> torch.Tensor:
    # (..., C, H, W)
    assert x.ndim >= 3, 'image must have 3 or more dimensions: (..., C, H, W)'
    assert x.dtype in {torch.float16, torch.float32, torch.float64, torch.complex32, torch.complex64}, f'unsupported dtype: {x.dtype}'
    # no gradient
    with torch.no_grad():
        # imaginary to real
        if x.dtype in {torch.complex32, torch.complex64}:
            x = torch.abs(x)
        # scale images
        if scale:
            m = x.min(dim=-3, keepdim=True).values.min(dim=-2, keepdim=True).values.min(dim=-1, keepdim=True).values
            M = x.max(dim=-3, keepdim=True).values.max(dim=-2, keepdim=True).values.max(dim=-1, keepdim=True).values
            x = (x - m) / (M - m)
        # move axis
        if move_channels:
            x = torch.moveaxis(x, -3, -1)
        # to uint8
        x = torch.clamp(x, 0, 1)
        x = (x * 255).to(torch.uint8)
    # done!
    x = x.detach()  # is this needeed?
    if to_cpu:
        x = x.cpu()
    return x


# ========================================================================= #
# Matplotlib Helper                                                         #
# ========================================================================= #


def plt_imshow(img, figsize=12, show=False, **kwargs):
    # check image shape
    assert img.ndim == 3
    assert img.shape[-1] in (1, 3)
    # figure size -- fixed width, adjust height according to image
    if isinstance(figsize, (int, str, Number)):
        size = np.array(img.shape[:2][::-1])
        figsize = tuple(size / size[0] * figsize)
    # create plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize, **kwargs)
    plt_hide_axis(ax)
    ax.imshow(img)
    fig.tight_layout()
    # done!
    if show:
        plt.show()
    return fig, ax


def _hide(hide, cond):
    assert hide in {True, False, 'all', 'edges', 'none'}
    return (hide is True) or (hide == 'all') or (hide == 'edges' and cond)


def plt_subplots(
    nrows: int = 1, ncols: int = 1,
    # custom
    title=None,
    titles=None,
    row_labels=None,
    col_labels=None,
    title_size: int = None,
    titles_size: int = None,
    label_size: int = None,
    hide_labels='edges',  # none, edges, all
    hide_axis='edges',    # none, edges, all
    # plt.subplots:
    sharex: str = False,
    sharey: str = False,
    subplot_kw=None,
    gridspec_kw=None,
    **fig_kw,
):
    assert isinstance(nrows, int)
    assert isinstance(ncols, int)
    # check titles
    if titles is not None:
        titles = np.array(titles)
        if titles.ndim == 1:
            titles = np.array([titles] + ([[None]*ncols] * (nrows-1)))
        assert titles.ndim == 2
    # get labels
    if (row_labels is None) or isinstance(row_labels, str):
        row_labels = [row_labels] * nrows
    if (col_labels is None) or isinstance(col_labels, str):
        col_labels = [col_labels] * ncols
    assert len(row_labels) == nrows, 'row_labels and nrows mismatch'
    assert len(col_labels) == ncols, 'row_labels and nrows mismatch'
    # check titles
    if titles is not None:
        assert len(titles) == nrows
        assert len(titles[0]) == ncols
    # create subplots
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, squeeze=False, subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **fig_kw)
    # generate
    for y in range(nrows):
        for x in range(ncols):
            ax = axs[y, x]
            plt_hide_axis(ax, hide_xaxis=_hide(hide_axis, y != nrows-1), hide_yaxis=_hide(hide_axis, x != 0))
            # modify ax
            if not _hide(hide_labels, y != nrows-1):
                ax.set_xlabel(col_labels[x], fontsize=label_size)
            if not _hide(hide_labels, x != 0):
                ax.set_ylabel(row_labels[y], fontsize=label_size)
            # set title
            if titles is not None:
                if titles[y][x] is not None:
                    ax.set_title(titles[y][x], fontsize=titles_size)
    # set title
    fig.suptitle(title, fontsize=title_size)
    # done!
    return fig, axs


def plt_subplots_imshow(
    grid,
    # custom:
    title=None,
    titles=None,
    row_labels=None,
    col_labels=None,
    title_size: int = None,
    titles_size: int = None,
    label_size: int = None,
    hide_labels='edges',  # none, edges, all
    hide_axis='all',    # none, edges, all
    # tight_layout:
    subplot_padding: Optional[float] = 1.08,
    # plt.subplots:
    sharex: str = False,
    sharey: str = False,
    subplot_kw=None,
    gridspec_kw=None,
    # imshow
    vmin: float = None,
    vmax: float = None,
    # extra
    show: bool = False,
    imshow_kwargs: dict = None,
    **fig_kw,
):
    # TODO: add automatic height & width
    fig, axs = plt_subplots(
        nrows=len(grid), ncols=len(grid[0]),
        # custom
        title=title,
        titles=titles,
        row_labels=row_labels,
        col_labels=col_labels,
        title_size=title_size,
        titles_size=titles_size,
        label_size=label_size,
        hide_labels=hide_labels,  # none, edges, all
        hide_axis=hide_axis,      # none, edges, all
        # plt.subplots:
        sharex=sharex,
        sharey=sharey,
        subplot_kw=subplot_kw,
        gridspec_kw=gridspec_kw,
        **fig_kw,
    )
    # show images
    for y, x in np.ndindex(axs.shape):
        axs[y, x].imshow(grid[y][x], vmin=vmin, vmax=vmax, **(imshow_kwargs if imshow_kwargs else {}))
    fig.tight_layout(**({} if (subplot_padding is None) else dict(pad=subplot_padding)))
    # done!
    if show:
        plt.show()
    return fig, axs


def plt_hide_axis(ax, hide_xaxis=True, hide_yaxis=True, hide_border=True, hide_axis_labels=False, hide_axis_ticks=True, hide_grid=True):
    if hide_xaxis:
        if hide_axis_ticks:
            ax.set_xticks([])
            ax.set_xticklabels([])
        if hide_axis_labels:
            ax.xaxis.label.set_visible(False)
    if hide_yaxis:
        if hide_axis_ticks:
            ax.set_yticks([])
            ax.set_yticklabels([])
        if hide_axis_labels:
            ax.yaxis.label.set_visible(False)
    if hide_border:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    if hide_grid:
        ax.grid(False)
    return ax


# ========================================================================= #
# Dataset Visualisation / Traversals                                        #
# ========================================================================= #


def visualize_dataset_traversal(
    dataset: DisentDataset,
    # inputs
    factor_names: Optional[NonNormalisedFactors] = None,
    num_frames: int = 9,
    seed: int = 777,
    base_factors=None,
    traverse_mode='cycle',
    # images & animations
    pad: int = 4,
    border: bool = True,
    bg_color: Number = None,
    # augment
    augment_fn: callable = None,
    data_mode: str = 'raw',
    # output
    output_wandb: bool = False,
):
    """
    Generic function that can return multiple parts of the dataset & factor traversal pipeline.
    - This only evaluates what is needed to compute the next components.
    - The returned grid, image and animation will always have 3 channels, RGB

    Tasks include:
        - factor_idxs
        - factors
        - grid
        - image
        - image_wandb
        - image_plt
        - animation
        - animation_wandb
    """

    # get factors from dataset
    factor_idxs = dataset.gt_data.normalise_factor_idxs(factor_names)

    # get factor traversals
    with TempNumpySeed(seed):
        factors = np.stack([
            dataset.gt_data.sample_random_factor_traversal(f_idx, base_factors=base_factors, num=num_frames, mode=traverse_mode)
            for f_idx in factor_idxs
        ], axis=0)

    # retrieve and augment image grid
    grid = [dataset.dataset_batch_from_factors(f, mode=data_mode) for f in factors]
    if augment_fn is not None:
        grid = [augment_fn(batch) for batch in grid]
    grid = np.stack(grid, axis=0)

    # TODO: this is kinda hacky, maybe rather add a check?
    # TODO: can this be moved into the `output_wandb` if statement?
    # - animations glitch out if they do not have 3 channels
    if grid.shape[-1] == 1:
        grid = grid.repeat(3, axis=-1)

    assert grid.ndim == 5
    assert grid.shape[-1] in (1, 3)

    # generate visuals
    image = make_image_grid(np.concatenate(grid, axis=0), pad=pad, border=border, bg_color=bg_color, num_cols=num_frames)
    animation = make_animated_image_grid(np.stack(grid, axis=0), pad=pad, border=border, bg_color=bg_color, num_cols=None)

    # convert to wandb
    if output_wandb:
        import wandb
        wandb_image = wandb.Image(image)
        wandb_animation = wandb.Video(np.transpose(animation, [0, 3, 1, 2]), fps=4, format='mp4')
        return (
            wandb_image,
            wandb_animation,
        )

    # return values
    return (
        grid,       # (FACTORS, NUM_FRAMES, H, W, C)
        image,      # ([[H+PAD]*[FACTORS+1]], [[W+PAD]*[NUM_FRAMES+1]], C)
        animation,  # (NUM_FRAMES, [H & FACTORS], [W & FACTORS], C) -- size is auto-chosen
    )


# ========================================================================= #
# 2d density plot                                                           #
# ========================================================================= #


def plt_2d_density(
    x,
    y,
    n_bins: int = 300,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
    ymin: Optional[float] = None,
    ymax: Optional[float] = None,
    ax: plt.Subplot = None,
    pcolormesh_kwargs: Optional[Dict[str, Any]] = None
):
    from scipy.stats import kde
    # https://www.python-graph-gallery.com/85-density-plot-with-matplotlib
    # convert inputs
    x = np.array(x)
    y = np.array(y)
    # prevent singular
    # x = np.random.randn(*x.shape) * (0.01 * max(x.max() - x.min(), 1))
    # y = np.random.randn(*y.shape) * (0.01 * max(y.max() - y.min(), 1))
    # get bounds
    if xmin is None: xmin = x.min()
    if xmax is None: xmax = x.max()
    if ymin is None: ymin = y.min()
    if ymax is None: ymax = y.max()
    # Evaluate a gaussian kde on a regular grid of nbins x nbins over data extents
    xi, yi = np.mgrid[xmin:xmax:n_bins*1j, ymin:ymax:n_bins*1j]
    try:
        k = kde.gaussian_kde([x, y])
        zi = k(np.stack([xi.flatten(), yi.flatten()], axis=0))
    except np.linalg.LinAlgError:
        log.warning('Could not create 2d_density plot')
        return
    # update args
    if ax is None: ax = plt
    if pcolormesh_kwargs is None: pcolormesh_kwargs = {}
    # Make the plot
    ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', **pcolormesh_kwargs)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
