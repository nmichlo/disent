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


from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

import research.util as H
from disent.dataset.data import GroundTruthData
from disent.dataset.util.state_space import StateSpace
from disent.util.strings.fmt import bytes_to_human


# ========================================================================= #
# Sub Dataset                                                               #
# ========================================================================= #


def _default_kwargs(kwargs):
    if kwargs is None:
        return {}
    return kwargs


# this is a hack to enable multi-threading
# its just easier than managing my processes manually!
# the wrapped function must take in indices of the values to be processed
# the new function when called creates a dataloader
def to_dataloader_maker(total: int, desc: str = None):
    def wrapper(func):
        def make_data_loader(batch_size: int = 1, num_workers: int = 0, prefetch_factor: int = 2, **dataloader_kwargs):
            class WrapperLoader(Dataset):
                def __getitem__(self, idx):
                    return func(idx)
                def __len__(self):
                    return total
            return tqdm(
                iterable=DataLoader(
                    WrapperLoader(),
                    shuffle=False,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    drop_last=False,
                    prefetch_factor=prefetch_factor,
                    **_default_kwargs(dataloader_kwargs),
                ),
                desc=desc,
                total=None if (total is None) else ((total + batch_size - 1) // batch_size),
            )
        return make_data_loader
    return wrapper


# ========================================================================= #
# Dataset Stats                                                             #
# ========================================================================= #


def factor_dist_matrix_shapes(gt_data: GroundTruthData) -> np.ndarray:
    # shape: (f_idx, num_factors + 1)
    return np.array([factor_dist_matrix_shape(gt_data=gt_data, f_idx=f_idx) for f_idx in range(gt_data.num_factors)])


def factor_dist_matrix_shape(gt_data: GroundTruthData, f_idx: int) -> Tuple[int, ...]:
    # using triangular matrices complicates everything
    # (np.prod(self._gt_data.factor_sizes) * self._gt_data.factor_sizes[i])               # symmetric, including diagonal in distance matrix
    # (np.prod(self._gt_data.factor_sizes) * (self._gt_data.factor_sizes[i] - 1)) // 2  # upper triangular matrix excluding diagonal
    # (np.prod(self._gt_data.factor_sizes) * (self._gt_data.factor_sizes[i] + 1)) // 2  # upper triangular matrix including diagonal
    return (*np.delete(gt_data.factor_sizes, f_idx), gt_data.factor_sizes[f_idx], gt_data.factor_sizes[f_idx])


def print_dist_matrix_stats(gt_data: GroundTruthData):
    # assuming storage as f32
    num_pairs = factor_dist_matrix_shapes(gt_data).prod(axis=1).sum(axis=0)
    pre_compute_bytes = num_pairs * (32 // 8)
    pairwise_compute_bytes = num_pairs * (32 // 8) * np.prod(gt_data.obs_shape) * 2
    traversal_compute_bytes = np.prod(gt_data.obs_shape) * np.prod(gt_data.factor_sizes) * gt_data.num_factors
    # string
    print(
        f'{f"{gt_data.name}:":12s} '
        f'{num_pairs:10d} (pairs) '
        f'{bytes_to_human(pre_compute_bytes):>22s} (pre-comp. f32) '
        f'{"x".join(str(i) for i in gt_data.img_shape):>11s} (obs. size)'
        f'{bytes_to_human(pairwise_compute_bytes):>22s} (comp. f32) '
        f'{bytes_to_human(traversal_compute_bytes):>22s} (opt. f32)'
    )


# ========================================================================= #
# Dataset Compute                                                           #
# ========================================================================= #


def _check_gt_data(gt_data: GroundTruthData):
    obs = gt_data[0]
    # checks
    assert isinstance(obs, torch.Tensor)
    assert obs.dtype == torch.float32


def compute_factor_dist_matrices(
    gt_data: GroundTruthData,
    f_idx: int,
    masked: bool = True,
    dataloader_kwargs: dict = None,
):
    _check_gt_data(gt_data)
    # load data
    f_states = StateSpace(factor_sizes=np.delete(gt_data.factor_sizes, f_idx))
    a_idxs, b_idxs = H.pair_indices_combinations(gt_data.factor_sizes[f_idx])
    # make dataloader
    @to_dataloader_maker(total=len(f_states), desc=f'{gt_data.name}: {f_idx}')
    @torch.no_grad()
    def compute_dist(idx: int):
        # get position
        base_pos = f_states.idx_to_pos(idx)
        base_factors = np.insert(base_pos, f_idx, 0)
        # get traversal: (f_size, H*W*C)
        traversal = [gt_data[i].flatten().numpy() for i in gt_data.iter_traversal_indices(f_idx=f_idx, base_factors=base_factors)]
        traversal = np.stack(traversal, axis=0)
        # compute mask & extract items
        if masked:
            mask = (traversal[0] != traversal[1])
            for item in traversal[2:]:
                mask |= (traversal[0] != item)
            traversal = traversal[:, mask]
        # compute distances
        dists = np.mean((traversal[a_idxs] - traversal[b_idxs]) ** 2, axis=1, dtype='float32')
        # return data
        return base_pos, dists
    # compute distance matrices
    f_dist_matrices = np.zeros(factor_dist_matrix_shape(gt_data=gt_data, f_idx=f_idx), dtype='float32')
    for batch_base_pos, batch_dists in compute_dist(**_default_kwargs(dataloader_kwargs)):
        for base_pos, dists in zip(batch_base_pos, batch_dists):
            f_dist_matrices[(*base_pos, a_idxs, b_idxs)] = dists
            f_dist_matrices[(*base_pos, b_idxs, a_idxs)] = dists
    # return distances
    return f_dist_matrices


def compute_all_factor_dist_matrices(
    gt_data: GroundTruthData,
    masked: bool = True,
    dataloader_kwargs: dict = None,
):
    """
    ALGORITHM:
        for each factor: O(num_factors)
            for each traversal: O(prod(<factor sizes excluding current factor>))
                for element in traversal: O(n)
                    -- compute overlapping mask
                    -- we use this mask to only transfer and compute over the needed data
                    -- we transfer the MASKED traversal to the GPU not the pairs
                for each pair in the traversal: O(n*(n-1)/2)  |  O(n**2)
                    -- compute each unique pairs distance
                    -- return distances
    """
    # for each factor, compute pairwise overlap
    all_dist_matrices = []
    for f_idx in range(gt_data.num_factors):
        f_dist_matrices = compute_factor_dist_matrices(
            gt_data=gt_data,
            f_idx=f_idx,
            masked=masked,
            dataloader_kwargs=dataloader_kwargs,
        )
        all_dist_matrices.append(f_dist_matrices)
    return all_dist_matrices


# ========================================================================= #
# TEST!                                                                     #
# ========================================================================= #


@torch.no_grad()
def main():
    for name in ['shapes3d', 'smallnorb', 'cars3d']:  # , 'shapes3d', 'dsprites', 'xysquares']:
        # get the dataset and delete the transform
        gt_data = H.make_data(name, transform_mode='float32')
        print_dist_matrix_stats(gt_data)
        f_dist_matrices = compute_all_factor_dist_matrices(
            gt_data,
            masked=True,
            dataloader_kwargs=dict(batch_size=1, num_workers=12))
        # plot distance matrices
        H.plt_subplots_imshow(
            grid=[[d.reshape([-1, *d.shape[-2:]]).mean(axis=0) for d in f_dist_matrices]],
            subplot_padding=0.5,
            figsize=(20, 10),
        )
        plt.show()


if __name__ == '__main__':
    main()
