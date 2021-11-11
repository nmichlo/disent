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

from functools import wraps
from typing import Optional
from typing import Sequence
from typing import TypeVar
from typing import Union

import numpy as np
from torch.utils.data import Dataset
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


_DO_COPY = object()


class DisentDataset(Dataset, LengthIter):


    def __init__(
        self,
        dataset: Union[Dataset, GroundTruthData],
        sampler: Optional[BaseDisentSampler] = None,
        transform=None,
        augment=None,
        return_indices: bool = False,
    ):
        super().__init__()
        # save attributes
        self._dataset = dataset
        self._sampler = SingleSampler() if (sampler is None) else sampler
        self._transform = transform
        self._augment = augment
        self._return_indices = return_indices
        # initialize sampler
        if not self._sampler.is_init:
            self._sampler.init(dataset)

    def shallow_copy(
        self,
        transform=_DO_COPY,
        augment=_DO_COPY,
        return_indices=_DO_COPY,
    ) -> 'DisentDataset':
        return DisentDataset(
            dataset=self._dataset,
            sampler=self._sampler,
            transform=self._transform if (transform is _DO_COPY) else transform,
            augment=self._augment if (augment is _DO_COPY) else augment,
            return_indices=self._return_indices if (return_indices is _DO_COPY) else return_indices,
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Properties                                                            #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def data(self) -> Dataset:
        return self._dataset

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
    def unwrapped_disent_dataset(self) -> 'DisentDataset':
        sampler = self._sampler.uninit_copy()
        assert type(sampler) is type(self._sampler)
        return DisentDataset(
            dataset=self.wrapped_data,
            sampler=sampler,
            transform=self._transform,
            augment=self._augment,
            return_indices=self._return_indices,
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
        # done!
        return obs

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Batches                                                               #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # TODO: default_collate should be replaced with a function
    #      that can handle tensors and nd.arrays, and return accordingly

    def dataset_batch_from_indices(self, indices: Sequence[int], mode: str):
        """Get a batch of observations X from a batch of factors Y."""
        return default_collate([self.dataset_get(idx, mode=mode) for idx in indices])

    def dataset_sample_batch(self, num_samples: int, mode: str, replace: bool = False, return_indices: bool = False):
        """Sample a batch of observations X."""
        # built in np.random.choice cannot handle large values: https://github.com/numpy/numpy/issues/5299#issuecomment-497915672
        indices = random_choice_prng(len(self), size=num_samples, replace=replace)
        # return batch
        batch = self.dataset_batch_from_indices(indices, mode=mode)
        # return values
        if return_indices:
            return batch, default_collate(indices)
        else:
            return batch

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Batches -- Ground Truth Only                                          #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @groundtruth_only
    def dataset_batch_from_factors(self, factors: np.ndarray, mode: str):
        """Get a batch of observations X from a batch of factors Y."""
        indices = self.gt_data.pos_to_idx(factors)
        return self.dataset_batch_from_indices(indices, mode=mode)

    @groundtruth_only
    def dataset_sample_batch_with_factors(self, num_samples: int, mode: str):
        """Sample a batch of observations X and factors Y."""
        factors = self.gt_data.sample_factors(num_samples)
        batch = self.dataset_batch_from_factors(factors, mode=mode)
        return batch, default_collate(factors)


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
