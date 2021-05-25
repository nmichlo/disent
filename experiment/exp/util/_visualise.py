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
from typing import Sequence

import numpy as np
import torch
from matplotlib import pyplot as plt


# ========================================================================= #
# images                                                                    #
# ========================================================================= #


# TODO: similar functions exist: output_image, to_img, to_imgs, reconstructions_to_images
def to_img(x: torch.Tensor, scale=False, to_cpu=True, move_channels=True):
    assert x.ndim == 3, 'image must have 3 dimensions: (C, H, W)'
    return to_imgs(x, scale=scale, to_cpu=to_cpu, move_channels=move_channels)


# TODO: similar functions exist: output_image, to_img, to_imgs, reconstructions_to_images
def to_imgs(x: torch.Tensor, scale=False, to_cpu=True, move_channels=True):
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
# images - show                                                             #
# ========================================================================= #


# TODO: replace this function
# TODO: similar functions exist: output_image, show_img, show_imgs
def show_img(x: torch.Tensor, scale=False, i=None, step=None, show=True, **kwargs):
    if show:
        if (i is None) or (step is None) or (i % step == 0):
            plt_imshow(img=to_img(x, scale=scale), **kwargs)
            plt.show()


# TODO: replace this function
# TODO: similar functions exist: output_image, show_img, show_imgs
def show_imgs(xs: Sequence[torch.Tensor], scale=False, i=None, step=None, show=True):
    if show:
        if (i is None) or (step is None) or (i % step == 0):
            w = int(np.ceil(np.sqrt(len(xs))))
            h = (len(xs) + w - 1) // w
            fig, axs = plt.subplots(h, w)
            for ax, im in zip(np.array(axs).flatten(), xs):
                ax.imshow(to_img(im, scale=scale))
                ax.set_axis_off()
            fig.tight_layout()
            plt.show()


# ========================================================================= #
# Matplotlib Helper                                                         #
# ========================================================================= #


def plt_imshow(img, figsize=12, **kwargs):
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
        titles = np.array(titles).reshape([nrows, ncols])
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
                ax.set_xlabel(col_labels[x])
            if not _hide(hide_labels, x != 0):
                ax.set_ylabel(row_labels[y])
            # set title
            if titles is not None:
                ax.set_title(titles[y][x])
    # set title
    fig.suptitle(title)
    # done!
    return fig, axs


def plt_subplots_imshow(
    grid,
    # custom:
    title=None,
    titles=None,
    row_labels=None,
    col_labels=None,
    hide_labels='edges',  # none, edges, all
    hide_axis='all',    # none, edges, all
    # tight_layout:
    subplot_padding=None,
    # plt.subplots:
    sharex: str = False,
    sharey: str = False,
    subplot_kw=None,
    gridspec_kw=None,
    **fig_kw,
):
    fig, axs = plt_subplots(
        nrows=len(grid), ncols=len(grid[0]),
        # custom
        title=title,
        titles=titles,
        row_labels=row_labels,
        col_labels=col_labels,
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
        axs[y, x].imshow(grid[y][x])
    fig.tight_layout(pad=subplot_padding)
    # done!
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
# END                                                                       #
# ========================================================================= #
