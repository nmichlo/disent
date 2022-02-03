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
import warnings
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Sized
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.utils.data

from disent.dataset import DisentDataset
from disent.dataset.data import Cars3dData
from disent.dataset.data import DSpritesData
from research.code.dataset.data import DSpritesImagenetData
from disent.dataset.data import GroundTruthData
from disent.dataset.data import Shapes3dData
from disent.dataset.data import SmallNorbData
from research.code.dataset.data import XColumnsData
from research.code.dataset.data import XYBlocksData
from disent.dataset.data import XYObjectData
from research.code.dataset.data import XYSquaresData
from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.sampling import GroundTruthSingleSampler
from disent.dataset.transform import Noop
from disent.dataset.transform import ToImgTensorF32
from disent.dataset.transform import ToImgTensorU8


# ========================================================================= #
# dataset io                                                                #
# ========================================================================= #


# TODO: this is much faster!
#
# import psutil
# import multiprocessing as mp
#
# def copy_batch_into(src: GroundTruthData, dst: torch.Tensor, i: int, j: int):
#     for k in range(i, min(j, len(dst))):
#         dst[k, ...] = src[k]
#     return (i, j)
#
# def load_dataset_into_memory(
#     gt_data: GroundTruthData,
#     workers: int = min(psutil.cpu_count(logical=False), 16),
# ) -> ArrayGroundTruthData:
#     # make data and tensors
#     tensor = torch.zeros(len(gt_data), *gt_data.obs_shape, dtype=gt_data[0].dtype).share_memory_()
#     # compute batch size
#     n = len(gt_data)
#     batch_size = (n + workers - 1) // workers
#     # load in batches
#     with mp.Pool(processes=workers) as POOL:
#         POOL.starmap(
#             copy_batch_into, [
#                 (gt_data, tensor, i, i + batch_size)
#                 for i in range(0, n, batch_size)
#             ]
#         )
#     # return array
#     return ArrayGroundTruthData.new_like(tensor, gt_data, array_chn_is_last=False)


def load_dataset_into_memory(gt_data: GroundTruthData, x_shape: Optional[Tuple[int, ...]] = None, batch_size=64, num_workers=min(os.cpu_count(), 16), dtype=torch.float32, raw_array=False):
    assert dtype in {torch.float16, torch.float32}
    # TODO: this should be part of disent?
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from disent.dataset.data import ArrayGroundTruthData
    # get observation shape
    # - manually specify this if the gt_data has a transform applied that resizes the observations for example!
    if x_shape is None:
        x_shape = gt_data.x_shape
    # load dataset into memory manually!
    data = torch.zeros(len(gt_data), *x_shape, dtype=dtype)
    # load all batches
    dataloader = DataLoader(gt_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    idx = 0
    for batch in tqdm(dataloader, desc='loading dataset into memory'):
        data[idx:idx+len(batch)] = batch.to(dtype)
        idx += len(batch)
    # done!
    if raw_array:
        return data
    else:
        # channels get swapped by the below ToImgTensorF32(), maybe allow `array_chn_is_last` as param
        return ArrayGroundTruthData.new_like(array=data, gt_data=gt_data, array_chn_is_last=False)


# ========================================================================= #
# dataset                                                                   #
# ========================================================================= #


TransformTypeHint = Union[Literal['uint8'], Literal['float'], Literal['float32'], Literal['none']]


def make_data(
    name: str = 'xysquares',
    factors: bool = False,
    data_root: str = 'data/dataset',
    try_in_memory: bool = False,
    load_into_memory: bool = False,
    load_memory_dtype: torch.dtype = torch.float16,
    transform_mode: TransformTypeHint = 'float32'
) -> GroundTruthData:
    # override values
    if load_into_memory and try_in_memory:
        warnings.warn('`load_into_memory==True` is incompatible with `try_in_memory==True`, setting `try_in_memory=False`!')
        try_in_memory = False
    # transform object
    TransformCls = {
        'uint8': ToImgTensorU8,
        'float32': ToImgTensorF32,
        'none': Noop,
    }[transform_mode]
    # make data
    if   name == 'xysquares':      data = XYSquaresData(transform=TransformCls())  # equivalent: [xysquares, xysquares_8x8, xysquares_8x8_s8]
    elif name == 'xysquares_1x1':  data = XYSquaresData(square_size=1, transform=TransformCls())
    elif name == 'xysquares_2x2':  data = XYSquaresData(square_size=2, transform=TransformCls())
    elif name == 'xysquares_4x4':  data = XYSquaresData(square_size=4, transform=TransformCls())
    elif name == 'xysquares_8x8':  data = XYSquaresData(square_size=8, transform=TransformCls())  # 8x8x8x8x8x8 = 262144  # equivalent: [xysquares, xysquares_8x8, xysquares_8x8_s8]
    elif name == 'xysquares_8x8_mini':  data = XYSquaresData(square_size=8, grid_spacing=14, transform=TransformCls())  # 5x5x5x5x5x5 = 15625
    # TOY DATASETS
    elif name == 'xysquares_8x8_toy':     data = XYSquaresData(square_size=8, grid_spacing=8, rgb=False, num_squares=1, transform=TransformCls())  # 8x8 = ?
    elif name == 'xysquares_8x8_toy_s1':  data = XYSquaresData(square_size=8, grid_spacing=1, rgb=False, num_squares=1, transform=TransformCls())  # ?x? = ?
    elif name == 'xysquares_8x8_toy_s2':  data = XYSquaresData(square_size=8, grid_spacing=2, rgb=False, num_squares=1, transform=TransformCls())  # ?x? = ?
    elif name == 'xysquares_8x8_toy_s4':  data = XYSquaresData(square_size=8, grid_spacing=4, rgb=False, num_squares=1, transform=TransformCls())  # ?x? = ?
    elif name == 'xysquares_8x8_toy_s8':  data = XYSquaresData(square_size=8, grid_spacing=8, rgb=False, num_squares=1, transform=TransformCls())  # 8x8 = ?
    # TOY DATASETS ALT
    elif name == 'xcolumns_8x_toy':     data = XColumnsData(square_size=8, grid_spacing=8, rgb=False, num_squares=1, transform=TransformCls())  # 8 = ?
    elif name == 'xcolumns_8x_toy_s1':  data = XColumnsData(square_size=8, grid_spacing=1, rgb=False, num_squares=1, transform=TransformCls())  # ? = ?
    elif name == 'xcolumns_8x_toy_s2':  data = XColumnsData(square_size=8, grid_spacing=2, rgb=False, num_squares=1, transform=TransformCls())  # ? = ?
    elif name == 'xcolumns_8x_toy_s4':  data = XColumnsData(square_size=8, grid_spacing=4, rgb=False, num_squares=1, transform=TransformCls())  # ? = ?
    elif name == 'xcolumns_8x_toy_s8':  data = XColumnsData(square_size=8, grid_spacing=8, rgb=False, num_squares=1, transform=TransformCls())  # 8 = ?
    # OVERLAPPING DATASETS
    elif name == 'xysquares_8x8_s1':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=1, transform=TransformCls())  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s2':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=2, transform=TransformCls())  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s3':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=3, transform=TransformCls())  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s4':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=4, transform=TransformCls())  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s5':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=5, transform=TransformCls())  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s6':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=6, transform=TransformCls())  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s7':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=7, transform=TransformCls())  # ?x?x?x?x?x? = ?
    elif name == 'xysquares_8x8_s8':  data = XYSquaresData(square_size=8, grid_size=8, grid_spacing=8, transform=TransformCls())  # 8x8x8x8x8x8 = 262144  # equivalent: [xysquares, xysquares_8x8, xysquares_8x8_s8]
    # OTHER SYNTHETIC DATASETS
    elif name == 'xyobject':  data = XYObjectData(transform=TransformCls())
    elif name == 'xyblocks':  data = XYBlocksData(transform=TransformCls())
    # NORMAL DATASETS
    elif name == 'cars3d':         data = Cars3dData(data_root=data_root,    prepare=True, transform=TransformCls(size=64))
    elif name == 'smallnorb':      data = SmallNorbData(data_root=data_root, prepare=True, transform=TransformCls(size=64))
    elif name == 'shapes3d':       data = Shapes3dData(data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    elif name == 'dsprites':       data = DSpritesData(data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    # CUSTOM DATASETS
    elif name == 'dsprites_imagenet_bg_100': data = DSpritesImagenetData(visibility=100, mode='bg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_80':  data = DSpritesImagenetData(visibility=80, mode='bg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_60':  data = DSpritesImagenetData(visibility=60, mode='bg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_40':  data = DSpritesImagenetData(visibility=40, mode='bg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_bg_20':  data = DSpritesImagenetData(visibility=20, mode='bg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    # --- #
    elif name == 'dsprites_imagenet_fg_100': data = DSpritesImagenetData(visibility=100, mode='fg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_80':  data = DSpritesImagenetData(visibility=80, mode='fg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_60':  data = DSpritesImagenetData(visibility=60, mode='fg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_40':  data = DSpritesImagenetData(visibility=40, mode='fg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    elif name == 'dsprites_imagenet_fg_20':  data = DSpritesImagenetData(visibility=20, mode='fg', data_root=data_root,  prepare=True, transform=TransformCls(), in_memory=try_in_memory)
    # DONE
    else: raise KeyError(f'invalid data name: {repr(name)}')
    # load into memory
    if load_into_memory:
        old_data, data = data, load_dataset_into_memory(data, dtype=load_memory_dtype, x_shape=(data.img_channels, 64, 64))
    # make dataset
    if factors:
        raise NotImplementedError('factor returning is not yet implemented in the rewrite! this needs to be fixed!')  # TODO!
    return data


def make_dataset(
    name: str = 'xysquares',
    factors: bool = False,
    data_root: str = 'data/dataset',
    try_in_memory: bool = False,
    load_into_memory: bool = False,
    load_memory_dtype: torch.dtype = torch.float16,
    transform_mode: TransformTypeHint = 'float32',
    sampler: BaseDisentSampler = None,
) -> DisentDataset:
    data = make_data(
        name=name,
        factors=factors,
        data_root=data_root,
        try_in_memory=try_in_memory,
        load_into_memory=load_into_memory,
        load_memory_dtype=load_memory_dtype,
        transform_mode=transform_mode,
    )
    return DisentDataset(
        data,
        sampler=GroundTruthSingleSampler() if (sampler is None) else sampler,
        return_indices=True
    )


def get_single_batch(dataloader, cuda=True):
    for batch in dataloader:
        (x_targ,) = batch['x_targ']
        break
    if cuda:
        x_targ = x_targ.cuda()
    return x_targ


# ========================================================================= #
# sampling helper                                                           #
# ========================================================================= #


# TODO: clean this up
def sample_factors(gt_data: GroundTruthData, num_obs: int = 1024, factor_mode: str = 'sample_random', factor: Union[int, str] = None):
    # sample multiple random factor traversals
    if factor_mode == 'sample_traversals':
        assert factor is not None, f'factor cannot be None when factor_mode=={repr(factor_mode)}'
        # get traversal
        f_idx = gt_data.normalise_factor_idx(factor)
        # generate traversals
        factors = []
        for i in range((num_obs + gt_data.factor_sizes[f_idx] - 1) // gt_data.factor_sizes[f_idx]):
            factors.append(gt_data.sample_random_factor_traversal(f_idx=f_idx))
        factors = np.concatenate(factors, axis=0)
    elif factor_mode == 'sample_random':
        factors = gt_data.sample_factors(num_obs)
    else:
        raise KeyError
    return factors


# TODO: move into dataset class
def sample_batch_and_factors(dataset: DisentDataset, num_samples: int, factor_mode: str = 'sample_random', factor: Union[int, str] = None, device=None):
    factors = sample_factors(dataset.gt_data, num_obs=num_samples, factor_mode=factor_mode, factor=factor)
    batch = dataset.dataset_batch_from_factors(factors, mode='target').to(device=device)
    factors = torch.from_numpy(factors).to(dtype=torch.float32, device=device)
    return batch, factors


# ========================================================================= #
# pair samplers                                                             #
# ========================================================================= #


def pair_indices_random(max_idx: int, approx_batch_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates pairs of indices in corresponding arrays,
    returning random permutations
    - considers [0, 1] and [1, 0] to be different  # TODO: consider them to be the same
    - never returns pairs with the same values, eg. [1, 1]
    - (default) number of returned values is: `max_idx * sqrt(max_idx) / 2`  -- arbitrarily chosen to scale slower than number of combinations
    """
    # defaults
    if approx_batch_size is None:
        approx_batch_size = int(max_idx * (max_idx ** 0.5) / 2)
    # sample values
    idx_a, idx_b = np.random.randint(0, max_idx, size=(2, approx_batch_size))
    # remove similar
    different = (idx_a != idx_b)
    idx_a = idx_a[different]
    idx_b = idx_b[different]
    # return values
    return idx_a, idx_b


def pair_indices_combinations(max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates pairs of indices in corresponding arrays,
    returning all combinations
    - considers [0, 1] and [1, 0] to be the same, only returns one of them
    - never returns pairs with the same values, eg. [1, 1]
    - number of returned values is: `max_idx * (max_idx-1) / 2`
    """
    # upper triangle excluding diagonal
    # - similar to: `list(itertools.combinations(np.arange(len(t_idxs)), 2))`
    idxs_a, idxs_b = np.triu_indices(max_idx, k=1)
    return idxs_a, idxs_b


def pair_indices_nearby(max_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates pairs of indices in corresponding arrays,
    returning nearby combinations
    - considers [0, 1] and [1, 0] to be the same, only returns one of them
    - never returns pairs with the same values, eg. [1, 1]
    - number of returned values is: `max_idx`
    """
    idxs_a = np.arange(max_idx)                # eg. [0 1 2 3 4 5]
    idxs_b = np.roll(idxs_a, shift=1, axis=0)  # eg. [1 2 3 4 5 0]
    return idxs_a, idxs_b


_PAIR_INDICES_FNS = {
    'random': pair_indices_random,
    'combinations': pair_indices_combinations,
    'nearby': pair_indices_nearby,
}


def pair_indices(max_idx: int, mode: str) -> Tuple[np.ndarray, np.ndarray]:
    try:
        fn = _PAIR_INDICES_FNS[mode]
    except:
        raise KeyError(f'invalid mode: {repr(mode)}')
    return fn(max_idx=max_idx)


# ========================================================================= #
# mask helper                                                               #
# ========================================================================= #


def make_changed_mask(batch: torch.Tensor, masked=True):
    if masked:
        mask = torch.zeros_like(batch[0], dtype=torch.bool)
        for i in range(len(batch)):
            mask |= (batch[0] != batch[i])
    else:
        mask = torch.ones_like(batch[0], dtype=torch.bool)
    return mask


# ========================================================================= #
# dataset indices                                                           #
# ========================================================================= #


def sample_unique_batch_indices(num_obs: int, num_samples: int) -> np.ndarray:
    assert num_obs >= num_samples, 'not enough values to sample'
    assert (num_obs - num_samples) / num_obs > 0.5, 'this method might be inefficient'
    # get random sample
    indices = set()
    while len(indices) < num_samples:
        indices.update(np.random.randint(low=0, high=num_obs, size=num_samples - len(indices)))
    # make sure indices are randomly ordered
    indices = np.fromiter(indices, dtype=int)
    # indices = np.array(list(indices), dtype=int)
    np.random.shuffle(indices)
    # return values
    return indices


def generate_epoch_batch_idxs(num_obs: int, num_batches: int, mode: str = 'shuffle') -> List[np.ndarray]:
    """
    Generate `num_batches` batches of indices.
    - Each index is in the range [0, num_obs).
    - If num_obs is not divisible by num_batches, then batches may not all be the same size.

    eg. [0, 1, 2, 3, 4] -> [[0, 1], [2, 3], [4]] -- num_obs=5, num_batches=3, sample_mode='range'
    eg. [0, 1, 2, 3, 4] -> [[1, 4], [2, 0], [3]] -- num_obs=5, num_batches=3, sample_mode='shuffle'
    eg. [0, 1, 0, 3, 2] -> [[0, 1], [0, 3], [2]] -- num_obs=5, num_batches=3, sample_mode='random'
    """
    # generate indices
    if mode == 'range':
        idxs = np.arange(num_obs)
    elif mode == 'shuffle':
        idxs = np.arange(num_obs)
        np.random.shuffle(idxs)
    elif mode == 'random':
        idxs = np.random.randint(0, num_obs, size=(num_obs,))
    else:
        raise KeyError(f'invalid mode={repr(mode)}')
    # return batches
    return np.array_split(idxs, num_batches)


def generate_epochs_batch_idxs(num_obs: int, num_epochs: int, num_epoch_batches: int, mode: str = 'shuffle') -> List[np.ndarray]:
    """
    Like generate_epoch_batch_idxs, but concatenate the batches of calling the function `num_epochs` times.
    - The total number of batches returned is: `num_epochs * num_epoch_batches`
    """
    batches = []
    for i in range(num_epochs):
        batches.extend(generate_epoch_batch_idxs(num_obs=num_obs, num_batches=num_epoch_batches, mode=mode))
    return batches


# ========================================================================= #
# Dataloader Sampler Utilities                                              #
# ========================================================================= #


class StochasticSampler(torch.utils.data.Sampler):
    """
    Sample random batches, not guaranteed to be unique or cover the entire dataset in one epoch!
    """

    def __init__(self, data_source: Union[Sized, int], batch_size: int = 128):
        super().__init__(data_source)
        if isinstance(data_source, int):
            self._len = data_source
        else:
            self._len = len(data_source)
        self._batch_size = batch_size
        assert isinstance(self._len, int)
        assert self._len > 0
        assert isinstance(self._batch_size, int)
        assert self._batch_size > 0

    def __iter__(self):
        while True:
            yield from np.random.randint(0, self._len, size=self._batch_size)


def yield_dataloader(dataloader: torch.utils.data.DataLoader, steps: int):
    i = 0
    while True:
        for it in dataloader:
            yield it
            i += 1
            if i >= steps:
                return


def StochasticBatchSampler(data_source: Union[Sized, int], batch_size: int):
    return torch.utils.data.BatchSampler(
        sampler=StochasticSampler(data_source=data_source, batch_size=batch_size),
        batch_size=batch_size,
        drop_last=True
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
