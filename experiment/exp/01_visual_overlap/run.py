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
from typing import Dict

import seaborn as sns
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm

import experiment.exp.util as H
from disent.data.groundtruth import *
from disent.dataset.groundtruth import GroundTruthDataset
from disent.transform import ToStandardisedTensor
from disent.util import to_numpy


# ========================================================================= #
# plot                                                                      #
# ========================================================================= #


def plot_overlap(a, b, mode='abs'):
    a, b = np.transpose(to_numpy(a), (1, 2, 0)), np.transpose(to_numpy(b), (1, 2, 0))
    if mode == 'binary':
        d = np.float32(a != b)
    elif mode == 'abs':
        d = np.abs(a - b)
    elif mode == 'diff':
        d = a - b
    else:
        raise KeyError
    d = (d - d.min()) / (d.max() - d.min())
    a, b, d = np.uint8(a * 255), np.uint8(b * 255), np.uint8(d * 255)
    fig, (ax_a, ax_b, ax_d) = plt.subplots(1, 3)
    ax_a.imshow(a)
    ax_b.imshow(b)
    ax_d.imshow(d)
    plt.show()


# ========================================================================= #
# CORE                                                                      #
# ========================================================================= #


def generate_data(gt_dataset, data_name: str, batch_size=64, samples=100_000, plot_diffs=False, load_cache=True, save_cache=True, overlap_loss: str = 'mse'):
    # cache
    file_path = os.path.join(os.path.dirname(__file__), f'cache/{data_name}_{samples}.pkl')
    if load_cache:
        if os.path.exists(file_path):
            print(f'loaded: {file_path}')
            return pd.read_pickle(file_path, compression='gzip')

    # generate
    with torch.no_grad():
        # dataframe
        df = defaultdict(lambda: defaultdict(list))

        # randomly overlapped data
        name = 'random'
        for i in tqdm(range((samples + (batch_size-1) - 1) // (batch_size-1)), desc=f'{data_name}: {name}'):
            # get random batch of unique elements
            idxs = H.sample_unique_batch_indices(num_obs=len(gt_dataset), num_samples=batch_size)
            batch = gt_dataset.dataset_batch_from_indices(idxs, mode='input')
            # plot
            if plot_diffs and (i == 0):
                plot_overlap(batch[0], batch[1])
            # store overlap results
            o = to_numpy(H.pairwise_overlap(batch[:-1], batch[1:], mode=overlap_loss))
            df[True][name].extend(o)
            df[False][name].extend(o)

        # traversal overlaps
        for f_idx in range(gt_dataset.num_factors):
            name = f'f_{gt_dataset.factor_names[f_idx]}'
            for i in tqdm(range((samples + (gt_dataset.factor_sizes[f_idx] - 1) - 1) // (gt_dataset.factor_sizes[f_idx] - 1)), desc=f'{data_name}: {name}'):
                # get random batch that is a factor traversal
                factors = gt_dataset.sample_random_factor_traversal(f_idx)
                batch = gt_dataset.dataset_batch_from_factors(factors, mode='input')
                # shuffle indices
                idxs = np.arange(len(factors))
                np.random.shuffle(idxs)
                # plot
                if plot_diffs and (i == 0): plot_overlap(batch[0], batch[1])
                # store overlap results
                df[True][name].extend(to_numpy(H.pairwise_overlap(batch[:-1], batch[1:], mode=overlap_loss)))
                df[False][name].extend(to_numpy(H.pairwise_overlap(batch[idxs[:-1]], batch[idxs[1:]], mode=overlap_loss)))

        # make dataframe!
        df = pd.DataFrame({
            'overlap':   [d         for ordered, data in df.items() for name, dat in data.items() for d in dat],
            'samples':   [name      for ordered, data in df.items() for name, dat in data.items() for d in dat],
            'ordered':   [ordered   for ordered, data in df.items() for name, dat in data.items() for d in dat],
            'data':      [data_name for ordered, data in df.items() for name, dat in data.items() for d in dat],
        })

    # save into cache
    if save_cache:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_pickle(file_path, compression='gzip')
        print(f'cached: {file_path}')

    return df


# ========================================================================= #
# plotting                                                                  #
# ========================================================================= #


def dual_plot_from_generated_data(df, data_name: str = None, save_name: str = None, tick_size: float = None, fig_l_pad=1, fig_w=7, fig_h=13):
    # make subplots
    cm = 1 / 2.54
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=((fig_l_pad+2*fig_w)*cm, fig_h*cm))
    if data_name is not None:
        fig.suptitle(data_name, fontsize=20)
    ax0.set_ylim(-0.025, 1.025)
    ax1.set_ylim(-0.025, 1.025)
    # plot
    ax0.set_title('Ordered Traversals')
    sns.ecdfplot(ax=ax0, data=df[df['ordered']==True], x="overlap", hue="samples")
    ax1.set_title('Shuffled Traversals')
    sns.ecdfplot(ax=ax1, data=df[df['ordered']==False], x="overlap", hue="samples")
    # edit plots
    ax0.set_xlabel('Overlap')
    ax1.set_xlabel('Overlap')
    if tick_size is not None:
        ax0.xaxis.set_major_locator(MultipleLocator(base=tick_size))
        ax1.xaxis.set_major_locator(MultipleLocator(base=tick_size))
    # ax0.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax0.set_ylabel('Cumulative Proportion')
    ax1.set_ylabel(None)
    ax1.set_yticklabels([])
    ax1.get_legend().remove()
    plt.tight_layout()
    # save
    if save_name is not None:
        path = os.path.join(os.path.dirname(__file__), 'plots', save_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f'saved: {path}')
    # show
    return fig


def all_plot_from_all_generated_data(dfs: dict, ordered=True, save_name: str = None, tick_sizes: Dict[str, float] = None, hide_extra_legends=False, fig_l_pad=1, fig_w=7, fig_h=13):
    if not dfs:
        return None
    # make subplots
    cm = 1 / 2.54
    fig, axs = plt.subplots(1, len(dfs), figsize=((fig_l_pad+len(dfs)*fig_w)*cm, fig_h * cm))
    axs = np.array(axs, dtype=np.object).reshape((-1,))
    # plot all
    for i, (ax, (data_name, df)) in enumerate(zip(axs, dfs.items())):
        # plot
        ax.set_title(data_name)
        sns.ecdfplot(ax=ax, data=df[df['ordered']==ordered], x="overlap", hue="samples")
        # edit plots
        ax.set_ylim(-0.025, 1.025)
        ax.set_xlabel('Overlap')
        if (tick_sizes is not None) and (data_name in tick_sizes):
            ax.xaxis.set_major_locator(MultipleLocator(base=tick_sizes[data_name]))
        if i == 0:
            ax.set_ylabel('Cumulative Proportion')
        else:
            if hide_extra_legends:
                ax.get_legend().remove()
            ax.set_ylabel(None)
            ax.set_yticklabels([])
    plt.tight_layout()
    # save
    if save_name is not None:
        path = os.path.join(os.path.dirname(__file__), 'plots', save_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f'saved: {path}')
    # show
    return fig


def plot_all(exp_name, datas, tick_sizes, samples: int, load=True, save=True, show_plt=True, show_dual_plt=False, save_plt=True, hide_extra_legends=False, fig_l_pad=1, fig_w=7, fig_h=13):
    # generate data and plot!
    dfs = {}
    for data_name, make_data_fn in datas.items():
        gt_dataset = GroundTruthDataset(make_data_fn(), transform=ToStandardisedTensor())
        df = generate_data(
            gt_dataset,
            data_name,
            batch_size=64,
            samples=samples,
            plot_diffs=False,
            load_cache=load,
            save_cache=save,
        )
        dfs[data_name] = df
        # plot ordered + shuffled
        fig = dual_plot_from_generated_data(
            df,
            data_name=data_name,
            save_name=f'{exp_name}/{data_name}_{samples}.png' if save_plt else None,
            tick_size=tick_sizes.get(data_name, None),
            fig_l_pad=fig_l_pad,
            fig_w=fig_w,
            fig_h=fig_h,
        )

        if show_dual_plt:
            plt.show()
        else:
            plt.close(fig)

    def _all_plot_generated(dfs, ordered: bool, suffix: str):
        fig = all_plot_from_all_generated_data(
            dfs,
            ordered=ordered,
            save_name=f'{exp_name}/{exp_name}-{"ordered" if ordered else "shuffled"}{suffix}.png' if save_plt else None,
            tick_sizes=tick_sizes,
            hide_extra_legends=hide_extra_legends,
            fig_l_pad=fig_l_pad,
            fig_w=fig_w,
            fig_h=fig_h,
        )
        if show_plt:
            plt.show()
        else:
            plt.close(fig)

    # all ordered plots
    _all_plot_generated(dfs, ordered=True, suffix='')
    _all_plot_generated({k: v for k, v in dfs.items() if k.lower().startswith('xy')}, ordered=True, suffix='-xy')
    _all_plot_generated({k: v for k, v in dfs.items() if not k.lower().startswith('xy')}, ordered=True, suffix='-normal')
    # all shuffled plots
    _all_plot_generated(dfs, ordered=False, suffix='')
    _all_plot_generated({k: v for k, v in dfs.items() if k.lower().startswith('xy')}, ordered=False, suffix='-xy')
    _all_plot_generated({k: v for k, v in dfs.items() if not k.lower().startswith('xy')}, ordered=False, suffix='-normal')
    # done!
    return dfs


def plot_dfs_stacked(dfs, title: str, save_name: str = None, show_plt=True, tick_size: float = None, fig_l_pad=1, fig_w=7, fig_h=13, **kwargs):
    # make new dataframe
    df = pd.concat((df[df['samples']=='random'] for df in dfs.values()))
    # make plot
    cm = 1 / 2.54
    fig, ax = plt.subplots(1, 1, figsize=((fig_l_pad+1*fig_w)*cm, fig_h*cm))
    ax.set_title(title)
    # plot
    # sns.kdeplot(ax=ax, data=df, x="overlap", hue="data", bw_adjust=2)
    sns.ecdfplot(ax=ax, data=df, x="overlap", hue="data")
    # edit settins
    # ax.set_ylim(-0.025, 1.025)
    ax.set_xlabel('Overlap')
    if tick_size is not None:
        ax.xaxis.set_major_locator(MultipleLocator(base=tick_size))
    ax.set_ylabel('Cumulative Proportion')
    plt.tight_layout()
    # save
    if save_name is not None:
        path = os.path.join(os.path.dirname(__file__), 'plots', save_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f'saved: {path}')
    # show
    if show_plt:
        plt.show()
    else:
        plt.close(fig)


def plot_unique_count(dfs, save_name: str = None, show_plt: bool = True, fig_l_pad=1, fig_w=1.5*7, fig_h=13):
    df_uniques = pd.DataFrame({
        'Grid Spacing': ['/'.join(data_name.split('-')[1:]) for data_name, df in dfs.items()],
        'Unique Overlap Values': [len(np.unique(df['overlap'].values, return_counts=True)[1]) for data_name, df in dfs.items()]
    })
    # make plot
    cm = 1 / 2.54
    fig, ax = plt.subplots(1, 1, figsize=((fig_l_pad+fig_w)*cm, fig_h*cm))
    ax.set_title('Increasing Overlap')
    sns.barplot(data=df_uniques, x='Grid Spacing', y='Unique Overlap Values')
    plt.gca().invert_xaxis()
    plt.tight_layout()
    # save
    if save_name is not None:
        path = os.path.join(os.path.dirname(__file__), 'plots', save_name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f'saved: {path}')
    # show
    if show_plt:
        plt.show()
    else:
        plt.close(fig)


# ========================================================================= #
# entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../gadfly.mplstyle'))

    # common settings
    SHARED_SETTINGS = dict(
        samples=50_000,
        load=True,
        save=True,
        show_plt=True,
        save_plt=True,
        show_dual_plt=False,
        fig_l_pad=1,
        fig_w=7,
        fig_h=13,
        tick_sizes={
            'DSprites': 0.05,
            'Shapes3d': 0.2,
            'Cars3d': 0.05,
            'XYSquares': 0.01,
            # increasing levels of overlap
            'XYSquares-1': 0.01,
            'XYSquares-2': 0.01,
            'XYSquares-3': 0.01,
            'XYSquares-4': 0.01,
            'XYSquares-5': 0.01,
            'XYSquares-6': 0.01,
            'XYSquares-7': 0.01,
            'XYSquares-8': 0.01,
            # increasing levels of overlap 2
            'XYSquares-1-8': 0.01,
            'XYSquares-2-8': 0.01,
            'XYSquares-3-8': 0.01,
            'XYSquares-4-8': 0.01,
            'XYSquares-5-8': 0.01,
            'XYSquares-6-8': 0.01,
            'XYSquares-7-8': 0.01,
            'XYSquares-8-8': 0.01,
        },
    )

    # EXPERIMENT 0 -- visual overlap on existing datasets

    dfs = plot_all(
        exp_name='dataset-overlap',
        datas={
          # 'XYObject':  lambda: XYObjectData(),
          # 'XYBlocks':  lambda: XYBlocksData(),
            'XYSquares': lambda: XYSquaresData(),
            'DSprites':  lambda: DSpritesData(),
            'Shapes3d':  lambda: Shapes3dData(),
            'Cars3d':    lambda: Cars3dData(),
          # 'SmallNorb': lambda: SmallNorbData(),
          # 'Mpi3d':     lambda: Mpi3dData(),
        },
        hide_extra_legends=False,
        **SHARED_SETTINGS
    )

    # EXPERIMENT 1 -- increasing visual overlap

    dfs = plot_all(
        exp_name='increasing-overlap',
        datas={
            'XYSquares-1': lambda: XYSquaresData(grid_spacing=1),
            'XYSquares-2': lambda: XYSquaresData(grid_spacing=2),
            'XYSquares-3': lambda: XYSquaresData(grid_spacing=3),
            'XYSquares-4': lambda: XYSquaresData(grid_spacing=4),
            'XYSquares-5': lambda: XYSquaresData(grid_spacing=5),
            'XYSquares-6': lambda: XYSquaresData(grid_spacing=6),
            'XYSquares-7': lambda: XYSquaresData(grid_spacing=7),
            'XYSquares-8': lambda: XYSquaresData(grid_spacing=8),
        },
        hide_extra_legends=True,
        **SHARED_SETTINGS
    )

    plot_unique_count(
        dfs=dfs,
        save_name='increasing-overlap/xysquares-increasing-overlap-counts.png',
    )

    plot_dfs_stacked(
        dfs=dfs,
        title='Increasing Overlap',
        exp_name='increasing-overlap',
        save_name='increasing-overlap/xysquares-increasing-overlap.png',
        tick_size=0.01,
        fig_w=13
    )

    # EXPERIMENT 2 -- increasing visual overlap fixed dim size

    dfs = plot_all(
        exp_name='increasing-overlap-fixed',
        datas={
            'XYSquares-1-8': lambda: XYSquaresData(square_size=8, grid_spacing=1, grid_size=8),
            'XYSquares-2-8': lambda: XYSquaresData(square_size=8, grid_spacing=2, grid_size=8),
            'XYSquares-3-8': lambda: XYSquaresData(square_size=8, grid_spacing=3, grid_size=8),
            'XYSquares-4-8': lambda: XYSquaresData(square_size=8, grid_spacing=4, grid_size=8),
            'XYSquares-5-8': lambda: XYSquaresData(square_size=8, grid_spacing=5, grid_size=8),
            'XYSquares-6-8': lambda: XYSquaresData(square_size=8, grid_spacing=6, grid_size=8),
            'XYSquares-7-8': lambda: XYSquaresData(square_size=8, grid_spacing=7, grid_size=8),
            'XYSquares-8-8': lambda: XYSquaresData(square_size=8, grid_spacing=8, grid_size=8),
        },
        hide_extra_legends=True,
        **SHARED_SETTINGS
    )

    plot_unique_count(
        dfs=dfs,
        save_name='increasing-overlap-fixed/xysquares-increasing-overlap-fixed-counts.png',
    )

    plot_dfs_stacked(
        dfs=dfs,
        title='Increasing Overlap',
        exp_name='increasing-overlap-fixed',
        save_name='increasing-overlap-fixed/xysquares-increasing-overlap-fixed.png',
        tick_size=0.01,
        fig_w=13
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

