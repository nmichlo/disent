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
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np
import os
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from tqdm import tqdm

from disent.metrics._flatness_components import compute_axis_score
from disent.metrics._flatness_components import compute_linear_score


# ========================================================================= #
# COPIED FROM EXP 1                                                         #
# ========================================================================= #
from disent.util import seed


def make_rel_path(*path_segments, is_file=True):
    assert not os.path.isabs(os.path.join(*path_segments)), 'path must be relative'
    path = os.path.join(os.path.dirname(__file__), *path_segments)
    folder_path = os.path.dirname(path) if is_file else path
    os.makedirs(folder_path, exist_ok=True)
    return path


def make_rel_path_add_ext(*path_segments, ext='.png'):
    # make path
    path = make_rel_path(*path_segments, is_file=True)
    if not os.path.splitext(path)[1]:
        path = f'{path}{ext}'
    return path


# ========================================================================= #
# distance function                                                         #
# ========================================================================= #


def _rotation_matrix(d, i, j, deg):
    assert 0 <= i < j <= d
    mat = torch.eye(d, dtype=torch.float32)
    r = np.deg2rad(deg)
    s, c = np.sin(r), np.cos(r)
    mat[i, i] = c
    mat[j, j] = c
    mat[j, i] = -s
    mat[i, j] = s
    return mat


def rotation_matrix_2d(deg):
    return _rotation_matrix(d=2, i=0, j=1, deg=deg)


def _random_rotation_matrix(d):
    mat = torch.eye(d, dtype=torch.float32)
    for i in range(d):
        for j in range(i+1, d):
            mat @= _rotation_matrix(d, i, j, np.random.randint(0, 360))
    return mat


def make_2d_line_points(n: int = 100, deg: float = 30, std_x: float = 1.0, std_y: float = 0.005):
    points = torch.randn(n, 2, dtype=torch.float32) * torch.as_tensor([[std_x, std_y]], dtype=torch.float32)
    points = points @ rotation_matrix_2d(deg)
    return points


def make_nd_line_points(n: int = 100, dims: int = 4, std_x: float = 1.0, std_y: float = 0.005):
    if not isinstance(dims, int):
        m, M = dims
        dims = np.randint(m, M)
    # generate numbers
    xs = torch.randn(n, dims, dtype=torch.float32)
    # axis standard deviations
    if isinstance(std_y, (float, int)):
        std_y = torch.full((dims-1,), fill_value=std_y, dtype=torch.float32)
    else:
        m, M = std_y
        std_y = torch.rand(dims-1, dtype=torch.float32) * (M - m) + m
    # scale axes
    std = torch.cat([torch.as_tensor([std_x]), std_y])
    xs = xs * std[None, :]
    # rotate
    return xs @ _random_rotation_matrix(dims)


def make_line_points(n: int = 100, deg: float = None, dims: int = 2, std_x: float = 1.0, std_y: float = 0.1):
    if deg is None:
        return make_nd_line_points(n=n, dims=dims, std_x=std_x, std_y=std_y)
    else:
        assert dims == 2, f'if "deg" is not None, then "dims" must equal 2, currently set to: {repr(dims)}'
        return make_2d_line_points(n=n, deg=deg, std_x=std_x, std_y=std_y)


# def random_line(std, n=100):
#     std = torch.as_tensor(std, dtype=torch.float32)
#     (d,) = std.shape
#     # generate numbers
#     xs = torch.randn(n, d, dtype=torch.float32)
#     # scale axes
#     xs = xs * std[None, :]
#     # rotate
#     return xs @ _random_rotation_matrix(d)


# ========================================================================= #
# GAUSSIAN                                                                  #
# ========================================================================= #


def gaussian_1d(x, s): return 1 / (np.sqrt(2 * np.pi) * s) * torch.exp(-(x**2)/(2*s**2))
def gaussian_1d_dx(x, s): return gaussian_1d(x, s) * (-x/s**2)
def gaussian_1d_dx2(x, s): return gaussian_1d(x, s) * ((x**2 - s**2)/s**4)


def gaussian_2d(x, y, sx, sy): return gaussian_1d(x, sx) * gaussian_1d(y, sy)
def gaussian_2d_dy(x, y, sx, sy): return gaussian_1d(x, sx) * gaussian_1d_dx(y, sy)
def gaussian_2d_dy2(x, y, sx, sy): return gaussian_1d(x, sx) * gaussian_1d_dx2(y, sy)


def rotated_radius_meshgrid(radius: float, num_points: int, deg: float = 0, device=None, return_orig=False) -> Tuple[torch.Tensor, torch.Tensor]:
    # x & y values centered around zero
    # p = torch.arange(size, device=device) - (size-1)/2
    p = torch.linspace(-radius, radius, num_points, device=device)
    x, y = torch.meshgrid(p, p)
    # matrix multiplication along first axis | https://pytorch.org/docs/stable/generated/torch.einsum.html
    rx, ry = torch.einsum('dxy,kd->kxy', torch.stack([x, y]), rotation_matrix_2d(deg))
    # result
    if return_orig:
        return (rx, ry), (x, y)
    return rx, ry


def rotated_guassian2d(sigma: Tuple[float, float], deg: float, trunc_sigma: Optional[float] = None, num_points: int = 511):
    sx, sy = sigma
    radius = (2.25*max(sx, sy)) if (trunc_sigma is None) else trunc_sigma
    (xs_r, ys_r), (xs, ys) = rotated_radius_meshgrid(radius=radius, num_points=num_points, deg=deg, return_orig=True)
    zs = gaussian_2d(xs_r, ys_r, sx=sx, sy=sy)
    zs /= zs.sum()
    return xs, ys, zs


def plot_gaussian(
    deg: float = 0.0,
    sigma=(1.0, 0.1),
    # contour
    contour_resolution: int = 255,
    contour_trunc_sigma: Optional[float] = None,
    contour_kwargs: Optional[dict] = None,
    # dots
    dots_num: Optional[int] = None,
    dots_kwargs: Optional[dict] = None,
    # axis
    ax=None,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca()
    # set limits
    trunc_sigma = (2.05 * max(sigma)) if (contour_trunc_sigma is None) else contour_trunc_sigma
    ax.set_xlim([-trunc_sigma, trunc_sigma])
    ax.set_ylim([-trunc_sigma, trunc_sigma])
    # plot contour
    xs, ys, zs = rotated_guassian2d(sigma=sigma, deg=deg, trunc_sigma=trunc_sigma, num_points=contour_resolution)
    ax.contourf(xs, ys, zs, **({} if contour_kwargs is None else contour_kwargs))
    # plot dots
    if dots_num is not None:
        points = make_line_points(n=dots_num, dims=2, deg=deg, std_x=sigma[0], std_y=sigma[1])
        ax.scatter(*points.T, **({} if dots_kwargs is None else dots_kwargs))
    # done
    return ax


# ========================================================================= #
# Generate Average Plots                                                    #
# ========================================================================= #


def score_grid(
    deg_rotations: Sequence[Optional[float]],
    y_std_ratios: Sequence[float],
    x_std: float = 1.0,
    num_points: int = 1000,
    num_dims: int = 2,
    use_std: bool = True,
    use_max: bool = False,
    norm: bool = True,
    return_points: bool = False,
):
    h, w = len(y_std_ratios), len(deg_rotations)
    # grids
    axis_scores   = torch.zeros([h, w], dtype=torch.float64)
    linear_scores = torch.zeros([h, w], dtype=torch.float64)
    if return_points:
        all_points    = torch.zeros([h, w, num_points, num_dims], dtype=torch.float64)
    # compute scores
    for i, y_std_ratio in enumerate(y_std_ratios):
        for j, deg in enumerate(deg_rotations):
            points = make_line_points(n=num_points, dims=num_dims, deg=deg, std_x=x_std, std_y=x_std * y_std_ratio)
            axis_scores[i, j] = compute_axis_score(points, use_std=use_std, use_max=use_max, norm=norm)
            linear_scores[i, j] = compute_linear_score(points, use_std=use_std, use_max=use_max, norm=norm)
            if return_points:
                all_points[i, j] = points
    # results
    if return_points:
        return axis_scores, linear_scores, all_points
    return axis_scores, linear_scores


def ave_score_grid(
    deg_rotations: Sequence[Optional[float]],
    y_std_ratios: Sequence[float],
    x_std: float = 1.0,
    num_points: int = 1000,
    num_dims: int = 2,
    use_std: bool = True,
    use_max: bool = False,
    norm: bool = True,
    repeats: int = 10,
):
    results = []
    # repeat
    for i in tqdm(range(repeats)):
        results.append(score_grid(deg_rotations=deg_rotations, y_std_ratios=y_std_ratios, x_std=x_std, num_points=num_points, num_dims=num_dims, use_std=use_std, use_max=use_max, norm=norm))
    # average results
    all_axis_scores, all_linear_scores = zip(*results)
    axis_scores   = torch.mean(torch.stack(all_axis_scores,   dim=0), dim=0)
    linear_scores = torch.mean(torch.stack(all_linear_scores, dim=0), dim=0)
    # results
    return axis_scores, linear_scores


def make_ave_scores_plot(
    std_num: int = 21,
    deg_num: int = 21,
    ndim: Optional[int] = None,
    # extra
    num_points: int = 1000,
    repeats: int = 25,
    x_std: float = 1.0,
    use_std: bool = True,
    use_max: bool = False,
    norm: bool = True,
    # cmap
    cmap_axis: str = 'GnBu_r',  # 'RdPu_r', 'GnBu_r', 'Blues_r', 'viridis', 'plasma', 'magma'
    cmap_linear: str = 'RdPu_r',  # 'RdPu_r', 'GnBu_r', 'Blues_r', 'viridis', 'plasma', 'magma'
    vertical: bool = True,
    # subplot settings
    subplot_size: float = 4.,
    subplot_padding: float = 1.5,
):
    # make sure to handle the random case
    deg_num = std_num if (ndim is None) else deg_num
    axis_scores, linear_scores = ave_score_grid(
        deg_rotations=np.linspace(0., 180., num=deg_num) if (ndim is None) else [None],
        y_std_ratios=np.linspace(0., 1., num=std_num),
        x_std=x_std,
        num_points=num_points,
        num_dims=2 if (ndim is None) else ndim,
        use_std=use_std,
        use_max=use_max,
        norm=norm,
        repeats=repeats,
    )
    # aspect ratio
    aspect_ratio = 180 * (std_num / deg_num)
    # vertical / horizontal settings
    if vertical:
        nrows, ncols = (2, 1)
        figsize = (subplot_size + 0.33, subplot_size * 2 * (deg_num / std_num) + 0.75)
    else:
        nrows, ncols = (1, 2)
        figsize = (subplot_size * 2 * (deg_num / std_num) + 0.75, subplot_size + 0.5)
    # make plot
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    (ax0, ax1) = axs
    # common axis settings
    def _normalise_axis(ax):
        ax.set_aspect(aspect_ratio)
        ax.set_xticks(np.linspace(0., 180., 5))
        ax.grid(False)
        ax.set_xlabel(f'θ - Rotation Degrees')
        ax.set_ylabel(f'σ - Standard Deviation')
    # linear subplot
    ax0.set_title('Linear Scores')
    ax0.imshow(linear_scores, cmap=cmap_linear, extent=[0., 180., 0., 1.])
    _normalise_axis(ax0)
    # axis subplot
    ax1.set_title('Axis Scores')
    ax1.imshow(axis_scores, cmap=cmap_axis, extent=[0., 180., 0., 1.])
    _normalise_axis(ax1)
    # general subplot settings
    fig.tight_layout(pad=subplot_padding)
    # plot_remove_inner_grid_ticks(axs.reshape(nrows, ncols), hide_axis_labels=True, hide_axis_ticks=False)
    return fig, axs


# ========================================================================= #
# Generate Grid Plots                                                       #
# ========================================================================= #


def plot_hide_axis(ax, hide_xaxis=True, hide_yaxis=True, hide_border=True, hide_axis_labels=False, hide_axis_ticks=True):
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
    return ax


def plot_scores(ax, axis_score, linear_score):
    assert 0 <= linear_score <= 1
    assert 0 <= axis_score <= 1

    linear_rgb = cm.get_cmap('RdPu_r')(np.clip(linear_score, 0., 1.))
    axis_rgb   = cm.get_cmap('GnBu_r')(np.clip(axis_score, 0., 1.))

    from matplotlib.lines import Line2D

    ax.legend(handles=[
        Line2D([0], [0], label=f'Linear: {float(linear_score):.2f}', color=linear_rgb, marker='o', markersize=10, linestyle='None'),
        Line2D([0], [0], label=f'Axis: {float(axis_score):.2f}',     color=axis_rgb,   marker='o', markersize=10, linestyle='None'),
    ])

    return ax


def plot_remove_inner_grid_ticks(axs, hide_axis_labels=False, hide_axis_ticks=True):
    if axs.ndim != 2:
        raise NotImplementedError('Only ndims == 2 is supported!')
    h, w = axs.shape
    for y, ax_row in enumerate(axs):
        for x, ax in enumerate(ax_row):
            plot_hide_axis(ax, hide_xaxis=(y != h-1), hide_yaxis=(x != 0), hide_border=True, hide_axis_labels=hide_axis_labels, hide_axis_ticks=hide_axis_ticks)


def make_grid_gaussian_score_plot(
    # grid
    y_std_ratios: Sequence[float] = (0.8, 0.2, 0.05)[::-1],  # (0.8, 0.4, 0.2, 0.1, 0.05),
    deg_rotations: Sequence[float] = (0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5),  # (0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165),
    # plot score options
    x_std: float = 1.0,
    # plot dot options
    dots_num: Optional[int] = None,
    # score options
    num_points: int = 10000,
    repeats: int = 100,
    use_std: bool = True,
    use_max: bool = False,
    norm: bool = True,
    # grid options
    subplot_size: float = 2.125,
    subplot_padding: float = 0.5,
    subplot_contour_kwargs: Optional[dict] = None,
    subplot_dots_kwargs: Optional[dict] = None,
):
    # defaults
    if subplot_contour_kwargs is None:
        subplot_contour_kwargs = dict(cmap='Blues')
    if subplot_dots_kwargs is None:
        subplot_dots_kwargs = dict(cmap='Purples')

    # make figure
    nrows, ncols = len(y_std_ratios), len(deg_rotations)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*subplot_size, nrows*subplot_size))

    # progress
    p = tqdm(total=axs.size, desc='generating_plot')

    # compute scores
    for y, y_std_ratio in enumerate(y_std_ratios):
        sigma = (x_std, x_std * y_std_ratio)
        # for each rotation
        for x, deg in enumerate(deg_rotations):
            # compute scores
            axis_score, linear_score = [], []
            for k in range(repeats):
                points = make_2d_line_points(n=num_points, deg=deg, std_x=sigma[0], std_y=sigma[1])
                axis_score.append(compute_axis_score(points, use_std=use_std, use_max=use_max, norm=norm))
                linear_score.append(compute_linear_score(points, use_std=use_std, use_max=use_max, norm=norm))
            axis_score, linear_score = np.mean(axis_score), np.mean(linear_score)
            # generate subplots
            ax = axs[y, x]
            plot_gaussian(deg=deg, sigma=sigma, dots_num=dots_num, ax=ax, contour_trunc_sigma=x_std * 2.05, contour_kwargs=subplot_contour_kwargs, dots_kwargs=subplot_dots_kwargs)
            plot_hide_axis(ax)
            plot_scores(ax, axis_score=axis_score, linear_score=linear_score)
            # modify ax
            if y == nrows-1:
                ax.set_xlabel(f'θ = {deg}°')
            if x == 0:
                ax.set_ylabel(f'σ = {sigma[1]}')
            # update progress
            p.update()

    # generate plot
    plt.tight_layout(pad=subplot_padding)
    return fig, axs


# ========================================================================= #
# MAIN                                                                      #
# ========================================================================= #


if __name__ == '__main__':
    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # plot everything
    seed(777)
    make_grid_gaussian_score_plot(repeats=250, num_points=25000)
    plt.savefig(make_rel_path_add_ext('plots/metric_grid', ext='.png'))
    plt.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    seed(777)
    make_ave_scores_plot(repeats=250, num_points=10000, use_max=False)
    plt.savefig(make_rel_path_add_ext('plots/metric_scores', ext='.png'))
    plt.show()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #




