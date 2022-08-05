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

import warnings
from functools import wraps
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import default_collate

from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.data import GroundTruthData
from disent.dataset.sampling import SingleSampler
from disent.dataset.wrapper import WrappedDataset
from disent.util.deprecate import deprecated
from disent.util.iters import LengthIter
from disent.util.math.random import random_choice_prng


# ========================================================================= #
# Helper                                                                    #
# -- Checking if the wrapped data is an instance of GroundTruthData adds    #
#    complexity, but it means the user doesn't have to worry about handling #
#    potentially different instances of the DisentDataset class             #
# ========================================================================= #


class NotGroundTruthDataError(Exception):
    """
    This error is thrown if the wrapped dataset is not GroundTruthData
    """


T = TypeVar('T')


def groundtruth_only(func: T) -> T:
    @wraps(func)
    def wrapper(self: 'DisentDataset', *args, **kwargs):
        if not self.is_ground_truth:
            raise NotGroundTruthDataError(f'Check `is_ground_truth` first before calling `{func.__name__}`, the dataset wrapped by {repr(self.__class__.__name__)} is not a {repr(GroundTruthData.__name__)}, instead got: {repr(self._dataset)}.')
        return func(self, *args, **kwargs)
    return wrapper


def wrapped_only(func):
    @wraps(func)
    def wrapper(self: 'DisentDataset', *args, **kwargs):
        if not self.is_wrapped_data:
            raise NotGroundTruthDataError(f'Check `is_data_wrapped` first before calling `{func.__name__}`, the dataset wrapped by {repr(self.__class__.__name__)} is not a {repr(WrappedDataset.__name__)}, instead got: {repr(self._dataset)}.')
        return func(self, *args, **kwargs)
    return wrapper


# ========================================================================= #
# Dataset Wrapper                                                           #
# ========================================================================= #


_REF_ = object()


class DisentDataset(Dataset, LengthIter):

    def __init__(
        self,
        dataset: Union[Dataset, GroundTruthData],  # TODO: this should be renamed to data
        sampler: Optional[BaseDisentSampler] = None,
        transform: Optional[callable] = None,
        augment: Optional[callable] = None,
        return_indices: bool = False,  # doesn't really hurt performance, might as well leave enabled by default?
        return_factors: bool = False,
    ):
        super().__init__()
        # save attributes
        self._dataset = dataset
        self._sampler = SingleSampler() if (sampler is None) else sampler
        self._transform = transform
        self._augment = augment
        self._return_indices = return_indices
        self._return_factors = return_factors
        # check sampler
        assert isinstance(self._sampler, BaseDisentSampler), f'{DisentDataset.__name__} got an invalid {BaseDisentSampler.__name__}: {type(self._sampler)}'
        # initialize sampler
        if not self._sampler.is_init:
            self._sampler.init(dataset)
        # warn if we are overriding a transform
        if self._transform is not None:
            if hasattr(dataset, '_transform') and dataset._transform:
                warnings.warn(f'{DisentDataset.__name__} has transform specified as well as wrapped dataset: {dataset}, are you sure this is intended?')
        # check the dataset if we are returning the factors
        if self._return_factors:
            assert isinstance(self._dataset, GroundTruthData), f'If `return_factors` is `True`, then the dataset must be an instance of: {GroundTruthData.__name__}, got: {type(dataset)}'

    def shallow_copy(
        self,
        dataset: Union[Dataset, GroundTruthData] =_REF_,  # TODO: this should be renamed to data
        sampler: Optional[BaseDisentSampler] = _REF_,
        transform: Optional[callable] = _REF_,
        augment: Optional[callable] = _REF_,
        return_indices: bool = _REF_,
        return_factors: bool = _REF_,
    ) -> 'DisentDataset':
        # instantiate shallow dataset copy, overwriting elements if specified
        return DisentDataset(
            dataset        = self._dataset               if (dataset is _REF_)        else dataset,
            sampler        = self._sampler.uninit_copy() if (sampler is _REF_)        else sampler,
            transform      = self._transform             if (transform is _REF_)      else transform,
            augment        = self._augment               if (augment is _REF_)        else augment,
            return_indices = self._return_indices        if (return_indices is _REF_) else return_indices,
            return_factors = self._return_factors        if (return_factors is _REF_) else return_factors,
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Properties                                                            #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def data(self) -> Dataset:
        return self._dataset

    @property
    def sampler(self) -> BaseDisentSampler:
        return self._sampler

    @property
    def transform(self) -> Optional[Callable[[object], object]]:
        return self._transform

    @property
    def augment(self) -> Optional[Callable[[object], object]]:
        return self._augment

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Ground Truth Only                                                     #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def is_ground_truth(self) -> bool:
        return isinstance(self._dataset, GroundTruthData)

    @property
    @deprecated('ground_truth_data property replaced with `gt_data`')
    @groundtruth_only
    def ground_truth_data(self) -> GroundTruthData:
        return self._dataset

    @property
    @groundtruth_only
    def gt_data(self) -> GroundTruthData:
        # TODO: deprecate this or the long version
        return self._dataset

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Wrapped Dataset                                                       #
    # -- TODO: this is a bit hacky                                          #
    # -- Allows us to compute disentanglement metrics over datasets         #
    #    derived from ground truth data                                     #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def is_wrapped_data(self):
        return isinstance(self._dataset, WrappedDataset)

    @property
    def is_wrapped_gt_data(self):
        return isinstance(self._dataset, WrappedDataset) and isinstance(self._dataset.data, GroundTruthData)

    @property
    @wrapped_only
    def wrapped_data(self):
        self._dataset: WrappedDataset
        return self._dataset.data

    @property
    @wrapped_only
    def wrapped_gt_data(self):
        self._dataset: WrappedDataset
        return self._dataset.gt_data

    @wrapped_only
    def unwrapped_shallow_copy(
        self,
        sampler: Optional[BaseDisentSampler] = _REF_,
        transform: Optional[callable] = _REF_,
        augment: Optional[callable] = _REF_,
        return_indices: bool = _REF_,
        return_factors: bool = _REF_,
    ) -> 'DisentDataset':
        # like shallow_copy, but unwrap the dataset instead!
        return self.shallow_copy(
            dataset=self.wrapped_data,
            sampler=sampler,
            transform=transform,
            augment=augment,
            return_indices=return_indices,
            return_factors=return_factors,
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Dataset                                                               #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        if self._sampler is not None:
            idxs = self._sampler(idx)
        else:
            idxs = (idx,)
        # get the observations
        return self._dataset_get_observation(*idxs)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Single Datapoints                                                     #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _datapoint_raw_to_target(self, dat):
        x_targ = dat
        if self._transform is not None:
            x_targ = self._transform(x_targ)
        return x_targ

    def _datapoint_target_to_input(self, x_targ):
        x = x_targ
        if self._augment is not None:
            x = self._augment(x)
            # some augmentations may convert a (C, H, W) to (1, C, H, W), undo this change
            # TODO: this should not be here! this should be handled by the user instead!
            x = _batch_to_observation(batch=x, obs_shape=x_targ.shape)
        return x

    def dataset_get(self, idx, mode: str):
        """
        Gets the specified datapoint, using the specified mode.
        - raw: direct untransformed/unaugmented observations
        - target: transformed observations
        - input: transformed then augmented observations
        - pair: (input, target) tuple of observations

        Pipeline:
            1. raw    = dataset[idx]
            2. target = transform(raw)
            3. input  = augment(target) = augment(transform(raw))

        :param idx: The index of the datapoint in the dataset
        :param mode: {'raw', 'target', 'input', 'pair'}
        :return: observation depending on mode
        """
        try:
            idx = int(idx)
        except:
            raise TypeError(f'Indices must be integer-like ({type(idx)}): {idx}')
        # we do not support indexing by lists
        x_raw = self._dataset[idx]
        # return correct data
        if mode == 'pair':
            x_targ = self._datapoint_raw_to_target(x_raw)  # applies self.transform
            x = self._datapoint_target_to_input(x_targ)    # applies self.augment
            return x, x_targ
        elif mode == 'input':
            x_targ = self._datapoint_raw_to_target(x_raw)  # applies self.transform
            x = self._datapoint_target_to_input(x_targ)    # applies self.augment
            return x
        elif mode == 'target':
            x_targ = self._datapoint_raw_to_target(x_raw)  # applies self.transform
            return x_targ
        elif mode == 'raw':
            return x_raw
        else:
            raise ValueError(f'Invalid {mode=}')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Multiple Datapoints                                                   #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _dataset_get_observation(self, *idxs):
        xs, xs_targ = zip(*(self.dataset_get(idx, mode='pair') for idx in idxs))
        # handle cases
        obs = {'x_targ': xs_targ}
        # 5-10% faster
        if self._augment is not None:
            obs['x'] = xs
        # add indices
        if self._return_indices:
            obs['idx'] = idxs
        # add factors
        if self._return_factors:
            # >>> this is about 10% faster than below, because we do not need to do conversions!
            obs['factors'] = tuple(np.array(np.unravel_index(idxs, self._dataset.factor_sizes)).T)
            # >>> builtin but slower method, does some magic for more than 2 dims, could replace with faster try_njit method, but then we need numba!
            # obs['factors1'] = tuple(self.gt_data.idx_to_pos(idxs))
        # done!
        return obs

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Batches                                                               #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # TODO: default_collate should be replaced with a function
    #      that can handle tensors and nd.arrays, and return accordingly

    def dataset_batch_from_indices(self, indices: Sequence[int], mode: str, collate: bool = True):
        """Get a batch of observations X from a batch of factors Y."""
        batch = [self.dataset_get(idx, mode=mode) for idx in indices]
        return default_collate(batch) if collate else batch

    def dataset_sample_batch(self, num_samples: int, mode: str, replace: bool = False, return_indices: bool = False, collate: bool = True, seed: Optional[int] = None):
        """Sample a batch of observations X."""
        # built in np.random.choice cannot handle large values: https://github.com/numpy/numpy/issues/5299#issuecomment-497915672
        indices = random_choice_prng(len(self._dataset), size=num_samples, replace=replace, seed=seed)
        # return batch
        batch = self.dataset_batch_from_indices(indices, mode=mode, collate=collate)
        # return values
        if return_indices:
            return batch, (default_collate(indices) if collate else indices)
        else:
            return batch

    def dataset_sample_elems(self, num_samples: int, mode: str, return_indices: bool = False, seed: Optional[int] = None):
        """Sample uncollated elements with replacement, like `dataset_sample_batch`"""
        return self.dataset_sample_batch(num_samples=num_samples, mode=mode, replace=True, return_indices=return_indices, collate=False, seed=seed)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Batches -- Ground Truth Only                                          #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # TODO: batches should be obtained from indices
    #       - the wrapped gt datasets should handle generating these indices, eg. factor traversals etc.

    @groundtruth_only
    def dataset_batch_from_factors(self, factors: np.ndarray, mode: str, collate: bool = True):
        """Get a batch of observations X from a batch of factors Y."""
        indices = self.gt_data.pos_to_idx(factors)
        return self.dataset_batch_from_indices(indices, mode=mode, collate=collate)

    @groundtruth_only
    def dataset_sample_batch_with_factors(self, num_samples: int, mode: str, collate: bool = True):
        """Sample a batch of observations X and factors Y."""
        factors = self.gt_data.sample_factors(num_samples)
        batch = self.dataset_batch_from_factors(factors, mode=mode, collate=collate)
        return batch, (default_collate(factors) if collate else factors)


class DisentIterDataset(IterableDataset, DisentDataset):

    # make sure we cannot obtain the length directly
    __len__ = None

    def __iter__(self):
        # this takes priority over __getitem__, otherwise __getitem__ would need to
        # raise an IndexError if out of bounds to signal the end of iteration
        while True:
            # yield the entire dataset
            # - repeating when it is done!
            yield from (self[i] for i in range(len(self._dataset)))


# ========================================================================= #
# util                                                                      #
# ========================================================================= #


def _batch_to_observation(batch, obs_shape):
    """
    Convert a batch of size 1, to a single observation.
    """
    if batch.shape != obs_shape:
        assert batch.shape == (1, *obs_shape), f'batch.shape={repr(batch.shape)} does not correspond to obs_shape={repr(obs_shape)} with batch dimension added'
        return batch.reshape(obs_shape)
    return batch


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
