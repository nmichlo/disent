#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import docs.examples.plotting_examples.util as H
from disent.dataset.data import GroundTruthData
from disent.dataset.data import Mpi3dData
from disent.dataset.data import SelfContainedHdf5GroundTruthData
from disent.dataset.transform import ToImgTensorF32
from disent.dataset.util.state_space import NonNormalisedFactorIdxs
from disent.dataset.util.stats import compute_data_mean_std
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.seeds import TempNumpySeed
from disent.util.visualize.vis_util import make_image_grid

# ========================================================================= #
# Factor Traversal Stats                                                    #
# ========================================================================= #


SampleModeHint = Union[Literal["random"], Literal["near"], Literal["combinations"]]


@torch.no_grad()
def sample_factor_traversal_info(
    gt_data: GroundTruthData,
    f_idx: Optional[int] = None,
    sample_mode: SampleModeHint = "random",
) -> dict:
    # load traversal
    factors, indices = gt_data.sample_random_factor_traversal(f_idx=f_idx, return_indices=True)
    obs = torch.stack([gt_data[i] for i in indices])  # TODO: this is the bottleneck! not threaded
    # get pairs
    idxs_a, idxs_b = H.pair_indices(max_idx=len(indices), mode=sample_mode)
    # compute deltas
    deltas = F.mse_loss(obs[idxs_a], obs[idxs_b], reduction="none").mean(dim=[-3, -2, -1]).numpy()
    fdists = np.abs(factors[idxs_a] - factors[idxs_b]).sum(axis=-1)  # (NUM, FACTOR_SIZE) -> (NUM,)
    # done!
    return dict(
        # traversals
        factors=factors,  # np.ndarray
        indices=indices,  # np.ndarray
        obs=obs,  # torch.Tensor
        # pairs
        idxs_a=idxs_a,  # np.ndarray
        idxs_b=idxs_b,  # np.ndarray
        deltas=deltas,  # np.ndarray
        fdists=fdists,  # np.ndarray
    )


def sample_factor_traversal_info_and_distmat(
    gt_data: GroundTruthData,
    f_idx: Optional[int] = None,
) -> dict:
    dat = sample_factor_traversal_info(gt_data=gt_data, f_idx=f_idx, sample_mode="combinations")
    # extract
    factors, idxs_a, idxs_b, deltas, fdists = dat["factors"], dat["idxs_a"], dat["idxs_b"], dat["deltas"], dat["fdists"]
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
        for _ in tqdm(range(num_traversal_sample), desc=f"{gt_data.name}: {factor_name}"):
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


def plot_traversal_stats(
    dataset_or_name: Union[str, GroundTruthData],
    num_repeats: int = 256,
    f_idxs: Optional[NonNormalisedFactorIdxs] = None,
    suffix: Optional[str] = None,
    save_path: Optional[str] = None,
    plot_title: Union[bool, str] = False,
    plt_scale: float = 6,
    col_titles: Union[bool, List[str]] = True,
    y_size_offset: float = 0.45,
    x_size_offset: float = 0.75,
    disable_labels: bool = False,
    bottom_labels: bool = False,
    label_size: int = 23,
    return_dists: bool = True,
):
    # - - - - - - - - - - - - - - - - - #

    def stats_fn(gt_data, i, f_idx):
        return sample_factor_traversal_info_and_distmat(gt_data=gt_data, f_idx=f_idx)

    grid_t = []
    grid_titles = []

    def plot_ax(stats: dict, i: int, f_idx: int):
        fdists_matrix = np.mean(stats["fdists_matrix"], axis=0)
        deltas_matrix = np.mean(stats["deltas_matrix"], axis=0)
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
        suptitle = f"{plot_title}"
    elif plot_title:
        suptitle = f'{gt_data.name} {f" {suffix}" if suffix else ""}'
    else:
        suptitle = None

    # generate plot
    _collect_stats_for_factors(
        gt_data=gt_data,
        f_idxs=f_idxs,
        stats_fn=stats_fn,
        keep_keys=["deltas", "fdists", "deltas_matrix", "fdists_matrix"],
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
        figsize=((1 / 2.54) * len(f_idxs) * plt_scale + x_size_offset, (1 / 2.54) * (2) * plt_scale + y_size_offset),
    )

    # recolor axes
    for ax0, ax1 in axs.T:
        ax0.images[0].set_cmap("Blues")
        ax1.images[0].set_cmap("Purples")

    fig.tight_layout()

    # save the path
    if save_path is not None:
        assert save_path.endswith(".png")
        ensure_parent_dir_exists(save_path)
        plt.savefig(save_path)
        print(f"saved {gt_data.name} to: {save_path}")

    # show it!
    plt.show()

    # - - - - - - - - - - - - - - - - - #
    if return_dists:
        return fig, grid_t, grid_titles
    return fig


# ========================================================================= #
# MAIN - DISTS                                                              #
# ========================================================================= #


@torch.no_grad()
def factor_stats(
    gt_data: GroundTruthData,
    f_idxs=None,
    min_samples: int = 100_000,
    min_repeats: int = 5000,
    recon_loss: str = "mse",
    sample_mode: str = "random",
) -> Tuple[Sequence[int], List[np.ndarray]]:
    from disent.frameworks.helper.reconstructions import ReconLossHandler
    from disent.registry import RECON_LOSSES

    recon_loss: ReconLossHandler = RECON_LOSSES[recon_loss](reduction="mean")

    f_dists = []
    f_idxs = gt_data.normalise_factor_idxs(f_idxs)
    # for each factor
    for f_idx in f_idxs:
        dists = []
        with tqdm(desc=gt_data.factor_names[f_idx], total=min_samples) as p:
            # for multiple random factor traversals along the factor
            while len(dists) < min_samples or p.n < min_repeats:
                # based on: sample_factor_traversal_info(...) # TODO: should add recon loss to that function instead
                factors, indices = gt_data.sample_random_factor_traversal(f_idx=f_idx, return_indices=True)
                obs = torch.stack([gt_data[i] for i in indices])
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


def get_random_dists(gt_data: GroundTruthData, num_samples: int = 100_000, recon_loss: str = "mse"):
    from disent.frameworks.helper.reconstructions import ReconLossHandler
    from disent.registry import RECON_LOSSES

    recon_loss: ReconLossHandler = RECON_LOSSES[recon_loss](reduction="mean")

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


def print_ave_dists(gt_data: GroundTruthData, num_samples: int = 100_000, recon_loss: str = "mse"):
    dists = get_random_dists(gt_data=gt_data, num_samples=num_samples, recon_loss=recon_loss)
    f_mean = np.mean(dists)
    f_std = np.std(dists)
    print(f"[{gt_data.name}] RANDOM ({len(gt_data)}, {len(dists)}) - mean: {f_mean:7.4f}  std: {f_std:7.4f}")


def print_ave_factor_stats(
    gt_data: GroundTruthData,
    f_idxs=None,
    min_samples: int = 100_000,
    min_repeats: int = 5000,
    recon_loss: str = "mse",
    sample_mode: str = "random",
):
    # compute average distances
    f_idxs, f_dists = factor_stats(
        gt_data=gt_data,
        f_idxs=f_idxs,
        min_repeats=min_repeats,
        min_samples=min_samples,
        recon_loss=recon_loss,
        sample_mode=sample_mode,
    )
    # compute dists
    f_means = [np.mean(d) for d in f_dists]
    f_stds = [np.std(d) for d in f_dists]
    # sort in order of importance
    order = np.argsort(f_means)[::-1]
    # print information
    for i in order:
        f_idx, f_mean, f_std = f_idxs[i], f_means[i], f_stds[i]
        print(
            f"[{gt_data.name}] {gt_data.factor_names[f_idx]} ({gt_data.factor_sizes[f_idx]}, {len(f_dists[f_idx])}) - mean: {f_mean:7.4f}  std: {f_std:7.4f}"
        )


def main_compute_dists(
    factor_samples: int = 50_000,
    min_repeats: int = 5000,
    random_samples: int = 50_000,
    recon_loss: str = "mse",
    sample_mode: str = "random",
    seed: int = 777,
):
    # plot standard datasets
    for name in [
        "dsprites",
        "shapes3d",
        "cars3d",
        "smallnorb",
        "xysquares_8x8_s8",
        "xyobject",
        "xyobject_shaded",
        "mpi3d_toy",
        "mpi3d_realistic",
        "mpi3d_real",
    ]:
        gt_data = H.make_data(name)
        if factor_samples is not None:
            with TempNumpySeed(seed):
                print_ave_factor_stats(
                    gt_data,
                    min_samples=factor_samples,
                    min_repeats=min_repeats,
                    recon_loss=recon_loss,
                    sample_mode=sample_mode,
                )
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

# [xy_object] y (25, 50000) - mean:  0.0116  std:  0.0139
# [xy_object] x (25, 50000) - mean:  0.0115  std:  0.0140
# [xy_object] color (24, 50000) - mean:  0.0088  std:  0.0084
# [xy_object] scale (5, 50000) - mean:  0.0051  std:  0.0056
# [xy_object] RANDOM (75000, 50000) - mean:  0.0145  std:  0.0111

# [xy_object] y (25, 50000) - mean:  0.0124  std:  0.0140
# [xy_object] x (25, 50000) - mean:  0.0122  std:  0.0144
# [xy_object] color (6, 50000) - mean:  0.0090  std:  0.0101
# [xy_object] scale (5, 50000) - mean:  0.0050  std:  0.0055
# [xy_object] intensity (4, 50000) - mean:  0.0033  std:  0.0039
# [xy_object] RANDOM (75000, 50000) - mean:  0.0145  std:  0.0111

# ========================================================================= #
# MAIN - PLOTTING                                                           #
# ========================================================================= #


def _grid_plot_save(path: str, imgs: Sequence[np.ndarray], show: bool = True):
    img = make_image_grid(imgs, pad=0, border=False, num_cols=-1)
    H.plt_imshow(img, show=True)
    imageio.imsave(path, img)


def _make_self_contained_dataset(h5_path):
    return SelfContainedHdf5GroundTruthData(h5_path=h5_path, transform=ToImgTensorF32())


def _print_data_mean_std(data_or_name, print_mean_std: bool = True):
    if print_mean_std:
        data = H.make_data(data_or_name) if isinstance(data_or_name, str) else data_or_name
        name = data_or_name if isinstance(data_or_name, str) else data.name
        mean, std = compute_data_mean_std(data)
        print(f"{name}\n    vis_mean: {mean.tolist()}\n    vis_std: {std.tolist()}")


def main_plotting(print_mean_std: bool = True):
    def sp(name):
        return os.path.join(os.path.dirname(__file__), "plots/dists", f"DIST_NO-FREQ_{name}.png")

    # plot xysquares with increasing overlap
    for s in [1, 2, 3, 4, 5, 6, 7, 8]:
        plot_traversal_stats(
            plt_scale=8,
            label_size=26,
            x_size_offset=0,
            y_size_offset=0.6,
            save_path=sp(f"xysquares_8x8_s{s}"),
            dataset_or_name=f"xysquares_8x8_s{s}",
            f_idxs=[1],
            col_titles=[f"Space: {s}px"],
        )
        _print_data_mean_std(f"xysquares_8x8_s{s}", print_mean_std)

    # plot xysquares with increasing overlap -- combined into one image
    _grid_plot_save(
        path=sp(f"xysquares_8x8_all"),
        imgs=[imageio.imread(sp(f"xysquares_8x8_s{s}"))[:, 2:-2, :3] for s in range(1, 9)],
    )
    _grid_plot_save(
        path=sp(f"xysquares_8x8_some"),
        imgs=[imageio.imread(sp(f"xysquares_8x8_s{s}"))[:, 2:-2, :3] for s in [1, 2, 4, 8]],
    )

    # replace the factor names!
    Mpi3dData.factor_names = ("color", "shape", "size", "elevation", "bg_color", "first_dof", "second_dof")

    # plot standard datasets
    for name in [
        "dsprites",
        "shapes3d",
        "cars3d",
        "smallnorb",
        "xyobject",
        "xyobject_shaded",
        "mpi3d_toy",
        "mpi3d_realistic",
        "mpi3d_real",
    ]:
        plot_traversal_stats(
            x_size_offset=0,
            y_size_offset=0.6,
            num_repeats=256,
            disable_labels=False,
            save_path=sp(name),
            dataset_or_name=name,
        )
        _print_data_mean_std(name, print_mean_std)

    # plot adversarial dsprites datasets
    for fg in [True, False]:
        for vis in [100, 75, 50, 25, 0]:
            name = f'dsprites_imagenet_{"fg" if fg else "bg"}_{vis}'
            plot_traversal_stats(save_path=sp(name), dataset_or_name=name, x_size_offset=0.4)
            _print_data_mean_std(name, print_mean_std)


# ========================================================================= #
# STATS                                                                     #
# ========================================================================= #


if __name__ == "__main__":
    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), "util/gadfly.mplstyle"))
    # run!
    main_plotting()
    main_compute_dists()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
