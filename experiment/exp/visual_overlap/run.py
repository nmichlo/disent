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
from typing import List
from typing import Union

import seaborn as sns
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

from disent.data.groundtruth import *
from disent.dataset.groundtruth import GroundTruthDataset
from disent.transform import ToStandardisedTensor
from disent.util import to_numpy


def sample_indices(high, num_samples) -> List[int]:
    assert high >= num_samples, 'not enough values to sample'
    assert (high - num_samples) / high > 0.5, 'this method might be inefficient'
    # get random sample
    indices = set()
    while len(indices) < num_samples:
        indices.update(np.random.randint(low=0, high=high, size=num_samples - len(indices)))
    # make sure indices are randomly ordered
    indices = np.array(list(indices), dtype=int)
    np.random.shuffle(indices)
    # return values
    return indices


def overlap(batch_a, batch_b):
    loss = -F.mse_loss(batch_a, batch_b, reduction='none').mean(dim=(-3, -2, -1))
    assert loss.ndim == 1
    return loss


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


DATAS = {
    'XYObject': lambda: XYObjectData(),
    'XYBlocks': lambda: XYBlocksData(),
    'XYSquares': lambda: XYSquaresData(),
    'DSprites': lambda: DSpritesData(),
    'Cars3d': lambda: Cars3dData(),
    'SmallNorb': lambda: SmallNorbData(),
    'Shapes3d': lambda: Shapes3dData(),
    'Mpi3d': lambda: Mpi3dData(),
}


def generate_data(data_name: str, batch_size=64, samples=100_000, plot_diffs=False, load_cache=True, save_cache=True):
    # cache
    file_path = os.path.join(os.path.dirname(__file__), f'cache/{data_name}_{samples}.pkl')
    if load_cache:
        if os.path.exists(file_path):
            print(f'loaded: {file_path}')
            return pd.read_pickle(file_path)

    # generate
    with torch.no_grad():
        # dataset vars
        data = DATAS[data_name]()
        dataset = GroundTruthDataset(data, transform=ToStandardisedTensor())

        # dataframe
        df = defaultdict(lambda: defaultdict(list))

        # randomly overlapped data
        name = 'random'
        for i in tqdm(range((samples + (batch_size-1) - 1) // (batch_size-1)), desc=f'{data_name}: {name}'):
            # get random batch of unique elements
            idxs = sample_indices(len(dataset), batch_size)
            batch = dataset.dataset_batch_from_indices(idxs, mode='input')
            # plot
            if plot_diffs and (i == 0): plot_overlap(batch[0], batch[1])
            # store overlap results
            o = to_numpy(overlap(batch[:-1], batch[1:]))
            df[True][name].extend(o)
            df[False][name].extend(o)

        # traversal overlaps
        for f_idx in range(dataset.num_factors):
            name = f'f_{dataset.factor_names[f_idx]}'
            for i in tqdm(range((samples + (dataset.factor_sizes[f_idx] - 1) - 1) // (dataset.factor_sizes[f_idx] - 1)), desc=f'{data_name}: {name}'):
                # get random batch that is a factor traversal
                factors = dataset.sample_random_traversal_factors(f_idx)
                batch = dataset.dataset_batch_from_factors(factors, mode='input')
                # shuffle indices
                idxs = np.arange(len(factors))
                np.random.shuffle(idxs)
                # plot
                if plot_diffs and (i == 0): plot_overlap(batch[0], batch[1])
                # store overlap results
                df[True][name].extend(to_numpy(overlap(batch[:-1], batch[1:])))
                df[False][name].extend(to_numpy(overlap(batch[idxs[:-1]], batch[idxs[1:]])))

        # make dataframe!
        df = pd.DataFrame({
            'overlap':  [d         for ordered, data in df.items() for name, dat in data.items() for d in dat],
            'samples':  [name      for ordered, data in df.items() for name, dat in data.items() for d in dat],
            'ordered':  [ordered   for ordered, data in df.items() for name, dat in data.items() for d in dat],
            'data':     [data_name for ordered, data in df.items() for name, dat in data.items() for d in dat],
        })

    # save into cache
    if save_cache:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_pickle(file_path)
        print(f'cached: {file_path}')

    return df


def dual_plot_from_generated_data(df, data: str = None, save: Union[str, bool] = True):
    # make subplots
    cm = 1 / 2.54
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(1+9*2*cm, 13*cm))
    if data is not None:
        fig.suptitle(data, fontsize=20)
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
    ax0.set_ylabel('Cumulative Proportion')
    ax1.set_ylabel(None)
    ax1.set_yticklabels([])
    ax1.get_legend().remove()
    plt.tight_layout()
    # save
    if save:
        name = save if isinstance(save, str) else f'overlap-cdf_{data}.png'
        path = os.path.join(os.path.dirname(__file__), 'plots', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f'saved: {path}')
    # show
    return fig


def all_plot_from_all_generated_data(dfs: dict, ordered=True, save: Union[str, bool] = True):
    # make subplots
    cm = 1 / 2.54
    fig, axs = plt.subplots(1, len(dfs), figsize=(1+9*cm*len(dfs), 13*cm))
    axs = np.array(axs, dtype=np.object).reshape((-1,))
    # plot all
    for i, (ax, (data, df)) in enumerate(zip(axs, dfs.items())):
        # plot
        ax.set_title(data)
        sns.ecdfplot(ax=ax, data=df[df['ordered']==ordered], x="overlap", hue="samples")
        # edit plots
        ax.set_ylim(-0.025, 1.025)
        ax.set_xlabel('Overlap')
        if i == 0:
            ax.set_ylabel('Cumulative Proportion')
        else:
            ax.set_ylabel(None)
            ax.set_yticklabels([])
    plt.tight_layout()
    # save
    if save:
        name = save if isinstance(save, str) else f'all-overlap-cdf_{"ordered" if ordered else "shuffled"}.png'
        path = os.path.join(os.path.dirname(__file__), 'plots', name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        print(f'saved: {path}')
    # show
    return fig


if __name__ == '__main__':

    plt.style.use(os.path.join(os.path.dirname(__file__), 'gadfly.mplstyle'))

    # generate data and plot!
    dfs = {}
    for data in ['DSprites', 'Cars3d', 'SmallNorb', 'Shapes3d', 'XYSquares']:
        df = generate_data(data, batch_size=64, samples=25000, plot_diffs=False)
        dfs[data] = df
        # plot ordered + shuffled
        dual_plot_from_generated_data(df, save=True)
        plt.show()

    # all ordered plots
    all_plot_from_all_generated_data(dfs, ordered=True, save='all-overlap-ordered.png')
    plt.show()
    all_plot_from_all_generated_data({k: v for k, v in dfs.items() if k.lower().startswith('xy')}, ordered=True, save='all-overlap-ordered_xy.png')
    plt.show()
    all_plot_from_all_generated_data({k: v for k, v in dfs.items() if not k.lower().startswith('xy')}, ordered=True, save='all-overlap-ordered_normal.png')
    plt.show()

    # all shuffled plots
    all_plot_from_all_generated_data(dfs, ordered=True, save='all-overlap-shuffled.png')
    plt.show()
    all_plot_from_all_generated_data({k: v for k, v in dfs.items() if k.lower().startswith('xy')}, ordered=True, save='all-overlap-shuffled_xy.png')
    plt.show()
    all_plot_from_all_generated_data({k: v for k, v in dfs.items() if not k.lower().startswith('xy')}, ordered=True, save='all-overlap-shuffled_normal.png')
    plt.show()
