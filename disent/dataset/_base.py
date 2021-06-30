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

from typing import final
from typing import List
from typing import Tuple

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import T_co

from disent.dataset.data.groundtruth import GroundTruthData
from disent.util.iters import LengthIter


# ========================================================================= #
# Base Sampler                                                              #
# ========================================================================= #


class DisentSampler(object):

    def __init__(self, num_samples: int):
        self._num_samples = num_samples

    @property
    def num_samples(self) -> int:
        return self._num_samples

    __initialized = False

    @final
    def init(self, dataset) -> 'DisentSampler':
        if self.__initialized:
            raise RuntimeError(f'Sampler: {repr(self.__class__.__name__)} has already been initialized, are you sure it is not being reused?')
        # initialize
        self.__initialized = True
        self._init(dataset)
        return self

    def _init(self, dataset):
        pass

    def __call__(self, idx: int) -> Tuple[int, ...]:
        raise NotImplementedError


# ========================================================================= #
# Base Dataset                                                              #
# ========================================================================= #


class DisentSamplingDataset(Dataset, LengthIter):

    # TODO: this can be simplified

    def __init__(self, dataset, sampler: DisentSampler, transform=None, augment=None):
        self._dataset = dataset
        self._sampler = sampler
        self._transform = transform
        self._augment = augment
        # initialize sampler
        self._sampler.init(dataset)
        # initialize
        super().__init__()

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx) -> T_co:
        idxs = self._sampler(idx)
        return self.dataset_get_observation(*idxs)

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

    def dataset_get_observation(self, *idxs):
        xs, xs_targ = zip(*(self.dataset_get(idx, mode='pair') for idx in idxs))
        # handle cases
        if self._augment is None:
            # makes 5-10% faster
            return {
                'x_targ': xs_targ,
            }
        else:
            return {
                'x': xs,
                'x_targ': xs_targ,
            }

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Batches                                                               #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def dataset_batch_from_indices(self, indices: List[int], mode: str):
        """Get a batch of observations X from a batch of factors Y."""
        return default_collate([self.dataset_get(idx, mode=mode) for idx in indices])

    def dataset_sample_batch(self, num_samples: int, mode: str):
        """Sample a batch of observations X."""
        # sample indices
        indices = set()
        while len(indices) < num_samples:
            indices.add(np.random.randint(0, len(self)))
        # done
        return self.dataset_batch_from_indices(sorted(indices), mode=mode)


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
# Ground Truth Dataset                                                      #
# ========================================================================= #


class DisentGroundTruthSamplingDataset(DisentSamplingDataset, GroundTruthData):

    def __init__(self, dataset, sampler: DisentSampler, transform=None, augment=None):
        assert isinstance(dataset, GroundTruthData), f'dataset is not an instance of {repr(GroundTruthData.__class__.__name__)}, got: {repr(dataset)}'
        # initialize parents -- this is a weird merge of classes, consider splitting this out?
        DisentSamplingDataset.__init__(self, dataset, sampler, transform=transform, augment=augment)
        GroundTruthData.__init__(self, transform=transform)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Single Datapoints                                                     #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self.dataset.factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self.dataset.factor_sizes

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self.dataset.observation_shape

    @final
    def _get_observation(self, idx):
        raise RuntimeError(f'`_get_observation` should never be called on instances of {repr(DisentGroundTruthSamplingDataset.__class__.__name__)}')

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Single Datapoints                                                     #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def dataset_batch_from_factors(self, factors: np.ndarray, mode: str):
        """Get a batch of observations X from a batch of factors Y."""
        return self.dataset_batch_from_indices(self.pos_to_idx(factors), mode=mode)

    def dataset_sample_batch_with_factors(self, num_samples: int, mode: str):
        """Sample a batch of observations X and factors Y."""
        factors = self.sample_factors(num_samples)
        batch = self.dataset_batch_from_factors(factors, mode=mode)
        return batch, default_collate(factors)

    def dataset_sample_batch(self, num_samples: int, mode: str):
        """Sample a batch of observations X."""
        factors = self.sample_factors(num_samples)
        batch = self.dataset_batch_from_factors(factors, mode=mode)
        return batch


# ========================================================================= #
# EXTRA                                                                     #
# ========================================================================= #


# class GroundTruthDatasetAndFactors(GroundTruthDataset):
#     def dataset_get_observation(self, *idxs):
#         return {
#             **super().dataset_get_observation(*idxs),
#             'factors': tuple(self.idx_to_pos(idxs))
#         }

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
