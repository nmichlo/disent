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
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt

import docs.examples.plotting_examples.util as H
from disent.dataset.data import Cars3d64Data
from disent.dataset.data import DSpritesData
from disent.dataset.data import GroundTruthData
from disent.dataset.data import Mpi3dData
from disent.dataset.data import Shapes3dData
from disent.dataset.data import SmallNorb64Data
from disent.dataset.data import XYObjectData
from disent.dataset.data import XYObjectShadedData
from disent.dataset.data import XYSquaresData
from disent.util.function import wrapped_partial
from disent.util.seeds import TempNumpySeed


# ========================================================================= #
# core                                                                      #
# ========================================================================= #


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.shape[-1] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    assert img.shape[-1] == 3, f'last channel of array is not of size 3 for RGB, got shape: {tuple(img.shape)}'
    return img


def plot_dataset_overlap(
    gt_data: GroundTruthData,
    f_idxs=None,
    obs_max: Optional[int] = None,
    obs_spacing: int = 1,
    rel_path=None,
    plot_base: bool = False,
    plot_combined: bool = True,
    plot_sidebar: bool = False,
    save=True,
    seed=777,
    plt_scale=4.5,
    offset=0.75,
):
    with TempNumpySeed(seed):
        # choose an f_idx
        f_idx = np.random.choice(gt_data.normalise_factor_idxs(f_idxs))
        f_name = gt_data.factor_names[f_idx]
        num_cols = gt_data.factor_sizes[f_idx]
        # get a traversal
        obs = [gt_data[i] for i in gt_data.pos_to_idx(gt_data.sample_random_factor_traversal(f_idx=f_idx))]
        # get subset
        if obs_max is not None:
            max_obs_spacing, i = obs_spacing, 1
            while max_obs_spacing*obs_max > len(obs):
                max_obs_spacing = obs_spacing-i
                i += 1
            i = max((len(obs) - obs_max*max_obs_spacing) // 2, 0)
            obs = obs[i:i+obs_max*obs_spacing:max_obs_spacing][:obs_max]
        # convert
        obs = np.array([ensure_rgb(x) for x in obs], dtype='float32') / 255
        # compute the distances
        grid = np.zeros([len(obs), len(obs), *obs[0].shape])
        for i, i_obs in enumerate(obs):
            for j, j_obs in enumerate(obs):
                grid[i, j] = np.abs(i_obs - j_obs)
        # normalize
        grid /= grid.max()

        # make figure
        factors, frames, _, _, c = grid.shape
        assert c == 3

        if plot_base:
            # plot
            fig, axs = H.plt_subplots_imshow(grid, label_size=18, title_size=24, title=f'{gt_data.name}: {f_name}', subplot_padding=None, figsize=(offset + (1/2.54)*frames*plt_scale, (1/2.54)*(factors+0.45)*plt_scale))
            # save figure
            if save and (rel_path is not None):
                path = H.make_rel_path_add_ext(rel_path, ext='.png')
                plt.savefig(path)
                print(f'saved: {repr(path)}')
            plt.show()

        if plot_combined:
            # add obs
            if True:
                factors += 1
                frames += 1
                # scaled_obs = obs
                scaled_obs = obs  * 0.5 + 0.25
                # grid = 1 - grid
                # grid = grid * 0.5 + 0.25
                grid = np.concatenate([scaled_obs[None, :], grid], axis=0)
                add_row = np.concatenate([np.ones_like(obs[0:1]), scaled_obs], axis=0)
                grid = np.concatenate([grid, add_row[:, None]], axis=1)
            # plot
            fig, axs = H.plt_subplots_imshow(grid, label_size=18, title_size=24, row_labels=["traversal"] + (["diff."] * len(obs)), col_labels=(["diff."] * len(obs)) + ["traversal"], title=f'{gt_data.name}: {f_name}', subplot_padding=None, figsize=(offset + (1/2.54)*frames*plt_scale, (1/2.54)*(factors+0.45)*plt_scale))
            # save figure
            if save and (rel_path is not None):
                path = H.make_rel_path_add_ext(rel_path + '__combined', ext='.png')
                plt.savefig(path)
                print(f'saved: {repr(path)}')
            plt.show()

        # plot
        if plot_sidebar:
            fig, axs = H.plt_subplots_imshow(obs[:, None], subplot_padding=None, figsize=(offset + (1/2.54)*1*plt_scale, (1/2.54)*(factors+0.45)*plt_scale))
            if save and (rel_path is not None):
                path = H.make_rel_path_add_ext(rel_path + '__v', ext='.png')
                plt.savefig(path)
                print(f'saved: {repr(path)}')
            plt.show()
            fig, axs = H.plt_subplots_imshow(obs[None, :], subplot_padding=None, figsize=(offset + (1/2.54)*frames*plt_scale, (1/2.54)*(1+0.45)*plt_scale))
            if save and (rel_path is not None):
                path = H.make_rel_path_add_ext(rel_path + '__h', ext='.png')
                plt.savefig(path)
                print(f'saved: {repr(path)}')
            plt.show()


# ========================================================================= #
# entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), 'util/gadfly.mplstyle'))

    # options
    all_squares = True
    add_random_traversal = True
    num_cols = 7
    seed = 48

    for gt_data_cls, name in [
        (wrapped_partial(XYSquaresData, grid_spacing=1, grid_size=8, no_warnings=True), f'xy-squares-spacing1'),
        (wrapped_partial(XYSquaresData, grid_spacing=2, grid_size=8, no_warnings=True), f'xy-squares-spacing2'),
        (wrapped_partial(XYSquaresData, grid_spacing=4, grid_size=8, no_warnings=True), f'xy-squares-spacing4'),
        (wrapped_partial(XYSquaresData, grid_spacing=8, grid_size=8, no_warnings=True), f'xy-squares-spacing8'),
    ]:
        plot_dataset_overlap(gt_data_cls(), rel_path=f'plots/overlap/overlap__{name}', obs_max=3, obs_spacing=4, seed=seed-40)

    # replace the factor names!
    Mpi3dData.factor_names = ('color', 'shape', 'size', 'elevation', 'bg_color', 'first_dof', 'second_dof')

    for gt_data_cls, name in [
        (XYObjectData,       f'xyobject'),
        (XYObjectShadedData, f'xyobject_shaded'),
        (DSpritesData,       f'dsprites'),
        (Shapes3dData,       f'shapes3d'),
        (Cars3d64Data,       f'cars3d'),
        (SmallNorb64Data,    f'smallnorb'),
        (wrapped_partial(Mpi3dData, in_memory=True, subset='toy'),       f'mpi3d_toy'),
        (wrapped_partial(Mpi3dData, in_memory=True, subset='realistic'), f'mpi3d_realistic'),
        (wrapped_partial(Mpi3dData, in_memory=True, subset='real'),      f'mpi3d_real'),
    ]:
        gt_data = gt_data_cls()
        for f_idx, f_name in enumerate(gt_data.factor_names):
            try:
                plot_dataset_overlap(gt_data, rel_path=f'plots/overlap/overlap__{name}__f{f_idx}-{f_name}', obs_max=3, obs_spacing=4, f_idxs=f_idx, seed=seed)
            except Exception as e:
                print('FAILED', gt_data_cls, f_idx, f_name, e)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
