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
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import research.util as H
from disent.dataset.data import GroundTruthData
from disent.dataset.data import SelfContainedHdf5GroundTruthData
from disent.dataset.util.state_space import NonNormalisedFactors
from disent.dataset.transform import ToImgTensorF32
from disent.dataset.util.stats import compute_data_mean_std
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.profiling import Timer
from disent.util.seeds import TempNumpySeed


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
    stats_fn: Callable[[GroundTruthData, int, int], Dict[str, Any]],
    keep_keys: Sequence[str],
    stats_callback: Optional[Callable[[Dict[str, List[Any]], int, int], None]] = None,
    return_stats: bool = True,
    num_traversal_sample: int = 100,
) -> List[Dict[str, List[Any]]]:
    # prepare
    f_idxs = gt_data.normalise_factor_idxs(f_idxs)
    # generate data per factor
    f_stats = []
    for i, f_idx in enumerate(f_idxs):
        factor_name = gt_data.factor_names[f_idx]
        factor_size = gt_data.factor_sizes[f_idx]
        # repeatedly generate stats per factor
        stats = defaultdict(list)
        for _ in tqdm(range(num_traversal_sample), desc=f'{gt_data.name}: {factor_name}'):
            data = stats_fn(gt_data, i, f_idx)
            for key in keep_keys:
                stats[key].append(data[key])
        # save factor stats
        if return_stats:
            f_stats.append(stats)
        if stats_callback:
            stats_callback(stats, i, f_idx)
    # done!
    if return_stats:
        return f_stats


# ========================================================================= #
# Plot Traversal Stats                                                      #
# ========================================================================= #


_COLORS = {
    'blue':   (None, 'Blues',   'Blues'),
    'red':    (None, 'Reds',    'Reds'),
    'purple': (None, 'Purples', 'Purples'),
    'green':  (None, 'Greens',  'Greens'),
    'orange': (None, 'Oranges', 'Oranges'),
}


def plot_traversal_stats(
    dataset_or_name: Union[str, GroundTruthData],
    num_repeats: int = 256,
    f_idxs: Optional[NonNormalisedFactors] = None,
    circular_distance: bool = False,
    color='blue',
    color_gt_dist='blue',
    color_im_dist='purple',
    suffix: Optional[str] = None,
    save_path: Optional[str] = None,
    plot_freq: bool = True,
    plot_title: Union[bool, str] = False,
    fig_block_size: float = 4.0,
    col_titles: Union[bool, List[str]] = True,
    hide_axis: bool = True,
    hide_labels: bool = True,
    y_size_offset: float = 0.0,
    x_size_offset: float = 0.0,
):
    # - - - - - - - - - - - - - - - - - #

    def stats_fn(gt_data, i, f_idx):
        return sample_factor_traversal_info_and_distmat(gt_data=gt_data, f_idx=f_idx, circular_distance=circular_distance)

    def plot_ax(stats: dict, i: int, f_idx: int):
        deltas = np.concatenate(stats['deltas'])
        fdists = np.concatenate(stats['fdists'])
        fdists_matrix = np.mean(stats['fdists_matrix'], axis=0)
        deltas_matrix = np.mean(stats['deltas_matrix'], axis=0)

        # ensure that if we limit the number of points, that we get good values
        with TempNumpySeed(777): np.random.shuffle(deltas)
        with TempNumpySeed(777): np.random.shuffle(fdists)

        # subplot!
        if plot_freq:
            ax0, ax1, ax2, ax3 = axs[:, i]
        else:
            (ax0, ax1), (ax2, ax3) = (None, None), axs[:, i]

        # get title
        curr_title = None
        if isinstance(col_titles, bool):
            if col_titles:
                curr_title = gt_data.factor_names[f_idx]
        else:
            curr_title = col_titles[i]

        # set column titles
        if curr_title is not None:
            (ax0 if plot_freq else ax2).set_title(f'{curr_title}\n', fontsize=24)

        # plot the frequency stuffs
        if plot_freq:
            ax0.violinplot([deltas], vert=False)
            ax0.set_xlabel('deltas')
            ax0.set_ylabel('proportion')

            ax1.set_title('deltas vs. fdists')
            ax1.scatter(x=deltas[:15_000], y=fdists[:15_000], s=20, alpha=0.1, c=c_points)
            H.plt_2d_density(
                x=deltas[:10_000], xmin=deltas.min(), xmax=deltas.max(),
                y=fdists[:10_000], ymin=fdists.min() - 0.5, ymax=fdists.max() + 0.5,
                n_bins=100,
                ax=ax1, pcolormesh_kwargs=dict(cmap=cmap_density, alpha=0.5),
            )
            ax1.set_xlabel('deltas')
            ax1.set_ylabel('fdists')

        # ax2.set_title('fdists')
        ax2.imshow(fdists_matrix, cmap=gt_cmap_img)
        if not hide_labels: ax2.set_xlabel('f_idx')
        if not hide_labels: ax2.set_ylabel('f_idx')
        if hide_axis: H.plt_hide_axis(ax2)

        # ax3.set_title('divergence')
        ax3.imshow(deltas_matrix, cmap=im_cmap_img)
        if not hide_labels: ax3.set_xlabel('f_idx')
        if not hide_labels: ax3.set_ylabel('f_idx')
        if hide_axis: H.plt_hide_axis(ax3)


    # - - - - - - - - - - - - - - - - - #

    # initialize
    gt_data: GroundTruthData = H.make_data(dataset_or_name) if isinstance(dataset_or_name, str) else dataset_or_name
    f_idxs = gt_data.normalise_factor_idxs(f_idxs)

    c_points, cmap_density, cmap_img = _COLORS[color]
    im_c_points, im_cmap_density, im_cmap_img = _COLORS[color if (color_im_dist is None) else color_im_dist]
    gt_c_points, gt_cmap_density, gt_cmap_img = _COLORS[color if (color_gt_dist is None) else color_gt_dist]

    n = 4 if plot_freq else 2

    # get additional spacing
    title_offset = 0 if (isinstance(col_titles, bool) and not col_titles) else 0.15

    # settings
    r, c = [n,  len(f_idxs)]
    h, w = [(n+title_offset)*fig_block_size + y_size_offset, len(f_idxs)*fig_block_size + x_size_offset]

    # initialize plot
    fig, axs = plt.subplots(r, c, figsize=(w, h), squeeze=False)

    if isinstance(plot_title, str):
        fig.suptitle(f'{plot_title}\n', fontsize=25)
    elif plot_title:
        fig.suptitle(f'{gt_data.name} [circular={circular_distance}]{f" {suffix}" if suffix else ""}\n', fontsize=25)

    # generate plot
    _collect_stats_for_factors(
        gt_data=gt_data,
        f_idxs=f_idxs,
        stats_fn=stats_fn,
        keep_keys=['deltas', 'fdists', 'deltas_matrix', 'fdists_matrix'],
        stats_callback=plot_ax,
        num_traversal_sample=num_repeats,
    )

    # finalize plot
    fig.tight_layout()  # (pad=1.4 if hide_labels else 1.08)

    # save the path
    if save_path is not None:
        assert save_path.endswith('.png')
        ensure_parent_dir_exists(save_path)
        plt.savefig(save_path)
        print(f'saved {gt_data.name} to: {save_path}')

    # show it!
    plt.show()

    # - - - - - - - - - - - - - - - - - #
    return fig


# TODO: fix
def plot_traversal_stats(
    dataset_or_name: Union[str, GroundTruthData],
    num_repeats: int = 256,
    f_idxs: Optional[NonNormalisedFactors] = None,
    circular_distance: bool = False,
    color='blue',
    color_gt_dist='blue',
    color_im_dist='purple',
    suffix: Optional[str] = None,
    save_path: Optional[str] = None,
    plot_freq: bool = True,
    plot_title: Union[bool, str] = False,
    plt_scale: float = 6,
    col_titles: Union[bool, List[str]] = True,
    hide_axis: bool = True,
    hide_labels: bool = True,
    y_size_offset: float = 0.45,
    x_size_offset: float = 0.75,
    disable_labels: bool = False,
    bottom_labels: bool = False,
    label_size: int = 23,
):
    # - - - - - - - - - - - - - - - - - #

    def stats_fn(gt_data, i, f_idx):
        return sample_factor_traversal_info_and_distmat(
            gt_data=gt_data, f_idx=f_idx, circular_distance=circular_distance
        )

    grid_t = []
    grid_titles = []

    def plot_ax(stats: dict, i: int, f_idx: int):
        fdists_matrix = np.mean(stats['fdists_matrix'], axis=0)
        deltas_matrix = np.mean(stats['deltas_matrix'], axis=0)
        grid_t.append([fdists_matrix, deltas_matrix])
        # get the title
        if isinstance(col_titles, bool):
            if col_titles:
                grid_titles.append(gt_data.factor_names[f_idx])
        else:
            grid_titles.append(col_titles[i])

    # initialize
    gt_data: GroundTruthData = H.make_data(dataset_or_name) if isinstance(dataset_or_name, str) else dataset_or_name
    f_idxs = gt_data.normalise_factor_idxs(f_idxs)

    # get title
    if isinstance(plot_title, str):
        suptitle = f'{plot_title}'
    elif plot_title:
        suptitle = f'{gt_data.name} [circular={circular_distance}]{f" {suffix}" if suffix else ""}'
    else:
        suptitle = None

    # generate plot
    _collect_stats_for_factors(
        gt_data=gt_data,
        f_idxs=f_idxs,
        stats_fn=stats_fn,
        keep_keys=['deltas', 'fdists', 'deltas_matrix', 'fdists_matrix'],
        stats_callback=plot_ax,
        num_traversal_sample=num_repeats,
    )

    labels = None
    if (not disable_labels) and grid_titles:
        labels = grid_titles

    # settings
    fig, axs = H.plt_subplots_imshow(
        grid=list(zip(*grid_t)),
        title=suptitle,
        titles=None if bottom_labels else labels,
        titles_size=label_size,
        col_labels=labels if bottom_labels else None,
        label_size=label_size,
        subplot_padding=None,
        figsize=((1/2.54) * len(f_idxs) * plt_scale + x_size_offset, (1/2.54) * (2) * plt_scale + y_size_offset)
    )

    # recolor axes
    for (ax0, ax1) in axs.T:
        ax0.images[0].set_cmap('Blues')
        ax1.images[0].set_cmap('Purples')

    fig.tight_layout()

    # save the path
    if save_path is not None:
        assert save_path.endswith('.png')
        ensure_parent_dir_exists(save_path)
        plt.savefig(save_path)
        print(f'saved {gt_data.name} to: {save_path}')

    # show it!
    plt.show()

    # - - - - - - - - - - - - - - - - - #
    return fig


# ========================================================================= #
# MAIN - DISTS                                                              #
# ========================================================================= #


@torch.no_grad()
def factor_stats(gt_data: GroundTruthData, f_idxs=None, min_samples: int = 100_000, min_repeats: int = 5000, recon_loss: str = 'mse', sample_mode: str = 'random') -> Tuple[Sequence[int], List[np.ndarray]]:
    from disent.registry import RECON_LOSSES
    from disent.frameworks.helper.reconstructions import ReconLossHandler
    recon_loss: ReconLossHandler = RECON_LOSSES[recon_loss](reduction='mean')

    f_dists = []
    f_idxs = gt_data.normalise_factor_idxs(f_idxs)
    # for each factor
    for f_idx in f_idxs:
        dists = []
        with tqdm(desc=gt_data.factor_names[f_idx], total=min_samples) as p:
            # for multiple random factor traversals along the factor
            while len(dists) < min_samples or p.n < min_repeats:
                # based on: sample_factor_traversal_info(...) # TODO: should add recon loss to that function instead
                factors, indices, obs = gt_data.sample_random_obs_traversal(f_idx=f_idx, obs_collect_fn=torch.stack)
                # random pairs -- we use this because it does not include [i == i]
                idxs_a, idxs_b = H.pair_indices(max_idx=len(indices), mode=sample_mode)
                # get distances
                d = recon_loss.compute_pairwise_loss(obs[idxs_a], obs[idxs_b])
                d = d.numpy().tolist()
                # H.plt_subplots_imshow([[np.moveaxis(o.numpy(), 0, -1) for o in obs]])
                # plt.show()
                dists.extend(d)
                p.update(len(d))
        # aggregate the average distances
        f_dists.append(np.array(dists)[:min_samples])

    return f_idxs, f_dists


def get_random_dists(gt_data: GroundTruthData, num_samples: int = 100_000, recon_loss: str = 'mse'):
    from disent.registry import RECON_LOSSES
    from disent.frameworks.helper.reconstructions import ReconLossHandler
    recon_loss: ReconLossHandler = RECON_LOSSES[recon_loss](reduction='mean')

    dists = []
    with tqdm(desc=gt_data.name, total=num_samples) as p:
        # for multiple random factor traversals along the factor
        while len(dists) < num_samples:
            # random pair
            i, j = np.random.randint(0, len(gt_data), size=2)
            # get distance
            d = recon_loss.compute_pairwise_loss(gt_data[i][None, ...], gt_data[j][None, ...])
            # plt.show()
            dists.append(float(d.flatten()))
            p.update()
    # done!
    return np.array(dists)


def print_ave_dists(gt_data: GroundTruthData, num_samples: int = 100_000, recon_loss: str = 'mse'):
    dists = get_random_dists(gt_data=gt_data, num_samples=num_samples, recon_loss=recon_loss)
    f_mean = np.mean(dists)
    f_std = np.std(dists)
    print(f'[{gt_data.name}] RANDOM ({len(gt_data)}, {len(dists)}) - mean: {f_mean:7.4f}  std: {f_std:7.4f}')


def print_ave_factor_stats(gt_data: GroundTruthData, f_idxs=None, min_samples: int = 100_000, min_repeats: int = 5000, recon_loss: str = 'mse', sample_mode: str = 'random'):
    # compute average distances
    f_idxs, f_dists = factor_stats(gt_data=gt_data, f_idxs=f_idxs, min_repeats=min_repeats, min_samples=min_samples, recon_loss=recon_loss, sample_mode=sample_mode)
    # compute dists
    f_means = [np.mean(d) for d in f_dists]
    f_stds = [np.std(d) for d in f_dists]
    # sort in order of importance
    order = np.argsort(f_means)[::-1]
    # print information
    for i in order:
        f_idx, f_mean, f_std = f_idxs[i], f_means[i], f_stds[i]
        print(f'[{gt_data.name}] {gt_data.factor_names[f_idx]} ({gt_data.factor_sizes[f_idx]}, {len(f_dists[f_idx])}) - mean: {f_mean:7.4f}  std: {f_std:7.4f}')


def main_compute_dists(factor_samples: int = 50_000, min_repeats: int = 5000, random_samples: int = 50_000, recon_loss: str = 'mse', sample_mode: str = 'random', seed: int = 777):
    # plot standard datasets
    for name in ['dsprites', 'shapes3d', 'cars3d', 'smallnorb', 'xysquares_8x8_s8']:
        gt_data = H.make_data(name)
        if factor_samples is not None:
            with TempNumpySeed(seed):
                print_ave_factor_stats(gt_data, min_samples=factor_samples, min_repeats=min_repeats, recon_loss=recon_loss, sample_mode=sample_mode)
        if random_samples is not None:
            with TempNumpySeed(seed):
                print_ave_dists(gt_data, num_samples=random_samples, recon_loss=recon_loss)

# [dsprites] position_y (32, 50000) - mean:  0.0584  std:  0.0378
# [dsprites] position_x (32, 50000) - mean:  0.0559  std:  0.0363
# [dsprites] scale (6, 50000) - mean:  0.0250  std:  0.0148
# [dsprites] shape (3, 50000) - mean:  0.0214  std:  0.0095
# [dsprites] orientation (40, 50000) - mean:  0.0172  std:  0.0106
# [dsprites] RANDOM (737280, 50000) - mean:  0.0754  std:  0.0289

# [3dshapes] wall_hue (10, 50000) - mean:  0.1122  std:  0.0661
# [3dshapes] floor_hue (10, 50000) - mean:  0.1086  std:  0.0623
# [3dshapes] object_hue (10, 50000) - mean:  0.0416  std:  0.0292
# [3dshapes] shape (4, 50000) - mean:  0.0207  std:  0.0161
# [3dshapes] scale (8, 50000) - mean:  0.0182  std:  0.0153
# [3dshapes] orientation (15, 50000) - mean:  0.0116  std:  0.0079
# [3dshapes] RANDOM (480000, 50000) - mean:  0.2432  std:  0.0918

# [cars3d] azimuth (24, 50000) - mean:  0.0355  std:  0.0185
# [cars3d] object_type (183, 50000) - mean:  0.0349  std:  0.0176
# [cars3d] elevation (4, 50000) - mean:  0.0174  std:  0.0100
# [cars3d] RANDOM (17568, 50000) - mean:  0.0519  std:  0.0188

# [smallnorb] lighting (6, 50000) - mean:  0.0531  std:  0.0563
# [smallnorb] category (5, 50000) - mean:  0.0113  std:  0.0066
# [smallnorb] rotation (18, 50000) - mean:  0.0090  std:  0.0071
# [smallnorb] instance (5, 50000) - mean:  0.0068  std:  0.0048
# [smallnorb] elevation (9, 50000) - mean:  0.0034  std:  0.0030
# [smallnorb] RANDOM (24300, 50000) - mean:  0.0535  std:  0.0529

# [xy_squares] y_B (8, 50000) - mean:  0.0104  std:  0.0000
# [xy_squares] x_B (8, 50000) - mean:  0.0104  std:  0.0000
# [xy_squares] y_G (8, 50000) - mean:  0.0104  std:  0.0000
# [xy_squares] x_G (8, 50000) - mean:  0.0104  std:  0.0000
# [xy_squares] y_R (8, 50000) - mean:  0.0104  std:  0.0000
# [xy_squares] x_R (8, 50000) - mean:  0.0104  std:  0.0000
# [xy_squares] RANDOM (262144, 50000) - mean:  0.0308  std:  0.0022

# ========================================================================= #
# MAIN - PLOTTING                                                           #
# ========================================================================= #


def _make_self_contained_dataset(h5_path):
    return SelfContainedHdf5GroundTruthData(h5_path=h5_path, transform=ToImgTensorF32())


def _print_data_mean_std(data_or_name, print_mean_std: bool = True):
    if print_mean_std:
        data = H.make_data(data_or_name) if isinstance(data_or_name, str) else data_or_name
        name = data_or_name if isinstance(data_or_name, str) else data.name
        mean, std = compute_data_mean_std(data)
        print(f'{name}\n    vis_mean: {mean.tolist()}\n    vis_std: {std.tolist()}')


def main_plotting(plot_all=False, print_mean_std=False):
    CIRCULAR = False
    PLOT_FREQ = False

    def sp(name):
        prefix = 'CIRCULAR_' if CIRCULAR else 'DIST_'
        prefix = prefix + ('FREQ_' if PLOT_FREQ else 'NO-FREQ_')
        return os.path.join(os.path.dirname(__file__), 'plots', f'{prefix}{name}.png')

    # plot xysquares with increasing overlap
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
        plot_traversal_stats(circular_distance=CIRCULAR, plt_scale=8, label_size=26, x_size_offset=0, y_size_offset=0.6, save_path=sp(f'xysquares_8x8_s{s}'), color='blue', dataset_or_name=f'xysquares_8x8_s{s}', f_idxs=[1], col_titles=[f'Space: {s}px'], plot_freq=PLOT_FREQ)
        _print_data_mean_std(f'xysquares_8x8_s{s}', print_mean_std)

    # plot standard datasets
    for name in ['dsprites', 'shapes3d', 'cars3d', 'smallnorb']:
        plot_traversal_stats(circular_distance=CIRCULAR, x_size_offset=0, y_size_offset=0.6, num_repeats=256, disable_labels=False, save_path=sp(name), color='blue', dataset_or_name=name, plot_freq=PLOT_FREQ)
        _print_data_mean_std(name, print_mean_std)

    if not plot_all:
        return

    # plot adversarial dsprites datasets
    for fg in [True, False]:
        for vis in [100, 80, 60, 40, 20]:
            name = f'dsprites_imagenet_{"fg" if fg else "bg"}_{vis}'
            plot_traversal_stats(circular_distance=CIRCULAR, save_path=sp(name), color='orange', dataset_or_name=name, plot_freq=PLOT_FREQ, x_size_offset=0.4)
            _print_data_mean_std(name, print_mean_std)

    BASE = os.path.abspath(os.path.join(__file__, '../../../out/adversarial_data_approx'))

    # plot adversarial datasets
    for color, folder in [
        # 'const' datasets
        ('purple', '2021-08-18--00-58-22_FINAL-dsprites_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('purple', '2021-08-18--01-33-47_FINAL-shapes3d_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('purple', '2021-08-18--02-20-13_FINAL-cars3d_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('purple', '2021-08-18--03-10-53_FINAL-smallnorb_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        # 'invert' datasets
        ('orange', '2021-08-18--03-52-31_FINAL-dsprites_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('orange', '2021-08-18--04-29-25_FINAL-shapes3d_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('orange', '2021-08-18--05-13-15_FINAL-cars3d_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        ('orange', '2021-08-18--06-03-32_FINAL-smallnorb_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06'),
        # stronger 'invert' datasets
        ('red', '2021-09-06--00-29-23_INVERT-VSTRONG-shapes3d_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06'),
        ('red', '2021-09-06--03-17-28_INVERT-VSTRONG-dsprites_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06'),
        ('red', '2021-09-06--05-42-06_INVERT-VSTRONG-cars3d_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06'),
        ('red', '2021-09-06--09-10-59_INVERT-VSTRONG-smallnorb_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06'),
    ]:
        data = _make_self_contained_dataset(f'{BASE}/{folder}/data.h5')
        plot_traversal_stats(circular_distance=CIRCULAR, save_path=sp(folder), color=color, dataset_or_name=data, plot_freq=PLOT_FREQ, x_size_offset=0.4)
        _print_data_mean_std(data, print_mean_std)


# ========================================================================= #
# STATS                                                                     #
# ========================================================================= #


if __name__ == '__main__':
    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))
    # run!
    # main_plotting()
    main_compute_dists()


# ========================================================================= #
# STATS                                                                     #
# ========================================================================= #


# 2021-08-18--00-58-22_FINAL-dsprites_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.04375297]
#     vis_std: [0.06837677]
# 2021-08-18--01-33-47_FINAL-shapes3d_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.48852729, 0.5872147 , 0.59863929]
#     vis_std: [0.08931785, 0.18920148, 0.23331079]
# 2021-08-18--02-20-13_FINAL-cars3d_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.88888636, 0.88274618, 0.87782785]
#     vis_std: [0.18967542, 0.20009377, 0.20805905]
# 2021-08-18--03-10-53_FINAL-smallnorb_self_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.74029344]
#     vis_std: [0.06706581]
#
# 2021-08-18--03-52-31_FINAL-dsprites_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.0493243]
#     vis_std: [0.09729655]
# 2021-08-18--04-29-25_FINAL-shapes3d_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.49514523, 0.58791172, 0.59616399]
#     vis_std: [0.08637031, 0.1895267 , 0.23397072]
# 2021-08-18--05-13-15_FINAL-cars3d_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.88851889, 0.88029857, 0.87666017]
#     vis_std: [0.200735 , 0.2151134, 0.2217553]
# 2021-08-18--06-03-32_FINAL-smallnorb_invert_margin_0.005_aw10.0_close_p_random_n_s50001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.73232105]
#     vis_std: [0.08755041]
#
# 2021-09-06--00-29-23_INVERT-VSTRONG-shapes3d_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.47992192, 0.51311111, 0.54627272]
#     vis_std: [0.28653814, 0.29201543, 0.27395435]
# 2021-09-06--03-17-28_INVERT-VSTRONG-dsprites_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.20482841]
#     vis_std: [0.33634909]
# 2021-09-06--05-42-06_INVERT-VSTRONG-cars3d_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.76418207, 0.75554032, 0.75075393]
#     vis_std: [0.31892905, 0.32751031, 0.33319886]
# 2021-09-06--09-10-59_INVERT-VSTRONG-smallnorb_invert_margin_0.05_aw10.0_same_k1_close_s200001_Adam_lr0.0005_wd1e-06
#     vis_mean: [0.69691603]
#     vis_std: [0.21310608]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
