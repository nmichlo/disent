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
from typing import List
from typing import Optional
from typing import Sequence
from typing import Sized
from typing import Tuple
from typing import Union

import numpy as np
import torch
from torch.utils.data import BatchSampler
from torch.utils.data import Sampler

from disent.dataset import DisentDataset
from disent.dataset.data import Cars3dData
from disent.dataset.data import GroundTruthData
from disent.dataset.data import Shapes3dData
from disent.dataset.data import SmallNorbData
from disent.dataset.data import XYSquaresData
from disent.dataset.sampling import GroundTruthSingleSampler
from disent.nn.transform import ToStandardisedTensor


# ========================================================================= #
# dataset                                                                   #
# ========================================================================= #


def _load_dataset_into_memory(gt_data: GroundTruthData, obs_shape: Tuple[int, ...], batch_size=64, num_workers=os.cpu_count() // 2, dtype=torch.float32):
    assert dtype in {torch.float16, torch.float32}
    # TODO: this should be part of disent?
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from disent.dataset.data import ArrayGroundTruthData
    # load dataset into memory manually!
    data = torch.zeros(len(gt_data), *obs_shape, dtype=dtype)
    # load all batches
    dataloader = DataLoader(gt_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    idx = 0
    for batch in tqdm(dataloader, desc='loading dataset into memory'):
        data[idx:idx+len(batch)] = batch.to(dtype)
        idx += len(batch)
    # done!
    return ArrayGroundTruthData.new_like(array=data, dataset=gt_data)


def make_dataset(name: str = 'xysquares', factors: bool = False, data_root='data/dataset', load_into_memory: bool = False, load_memory_dtype=torch.float16) -> DisentDataset:
    # make data
    if   name == 'xysquares':      data = XYSquaresData(transform=ToStandardisedTensor())
    elif name == 'xysquares_1x1':  data = XYSquaresData(square_size=1, transform=ToStandardisedTensor())
    elif name == 'xysquares_2x2':  data = XYSquaresData(square_size=2, transform=ToStandardisedTensor())
    elif name == 'xysquares_4x4':  data = XYSquaresData(square_size=4, transform=ToStandardisedTensor())
    elif name == 'xysquares_8x8':  data = XYSquaresData(square_size=8, transform=ToStandardisedTensor())
    elif name == 'cars3d':         data = Cars3dData(data_root=data_root,    prepare=True, transform=ToStandardisedTensor(size=64))
    elif name == 'smallnorb':      data = SmallNorbData(data_root=data_root, prepare=True, transform=ToStandardisedTensor(size=64))
    elif name == 'shapes3d':       data = Shapes3dData(data_root=data_root,  prepare=True, transform=ToStandardisedTensor())
    else: raise KeyError(f'invalid data name: {repr(name)}')
    # load into memory
    if load_into_memory:
        data = _load_dataset_into_memory(data, obs_shape=(3, 64, 64), dtype=load_memory_dtype)
    # make dataset
    if factors:
        raise NotImplementedError('factor returning is not yet implemented in the rewrite! this needs to be fixed!')  # TODO!
    return DisentDataset(data, sampler=GroundTruthSingleSampler())


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


def normalise_factor_idx(dataset: GroundTruthData, factor: Union[int, str]) -> int:
    if isinstance(factor, str):
        try:
            f_idx = dataset.factor_names.index(factor)
        except:
            raise KeyError(f'{repr(factor)} is not one of: {dataset.factor_names}')
    else:
        f_idx = factor
    assert isinstance(f_idx, (int, np.int32, np.int64, np.uint8))
    assert 0 <= f_idx < dataset.num_factors
    return int(f_idx)


# general type
NonNormalisedFactors = Union[Sequence[Union[int, str]], Union[int, str]]


def normalise_factor_idxs(gt_data: GroundTruthData, factors: NonNormalisedFactors) -> np.ndarray:
    if isinstance(factors, (int, str)):
        factors = [factors]
    factors = np.array([normalise_factor_idx(gt_data, factor) for factor in factors])
    assert len(set(factors)) == len(factors)
    return factors


def get_factor_idxs(gt_data: GroundTruthData, factors: Optional[NonNormalisedFactors] = None) -> np.ndarray:
    if factors is None:
        return np.arange(gt_data.num_factors)
    return normalise_factor_idxs(gt_data, factors)


# TODO: clean this up
def sample_factors(gt_data: GroundTruthData, num_obs: int = 1024, factor_mode: str = 'sample_random', factor: Union[int, str] = None):
    # sample multiple random factor traversals
    if factor_mode == 'sample_traversals':
        assert factor is not None, f'factor cannot be None when factor_mode=={repr(factor_mode)}'
        # get traversal
        f_idx = normalise_factor_idx(gt_data, factor)
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
    factors = sample_factors(dataset.ground_truth_data, num_obs=num_samples, factor_mode=factor_mode, factor=factor)
    batch = dataset.dataset_batch_from_factors(factors, mode='target').to(device=device)
    factors = torch.from_numpy(factors).to(dtype=torch.float32, device=device)
    return batch, factors


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


class StochasticSampler(Sampler):
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
    return BatchSampler(
        sampler=StochasticSampler(data_source=data_source, batch_size=batch_size),
        batch_size=batch_size,
        drop_last=True
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
