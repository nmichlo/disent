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

from disent.dataset.data import GroundTruthData
from disent.dataset.data._raw import Hdf5Dataset


class TransformDataset(GroundTruthData):

    # TODO: all data should be datasets
    # TODO: file preparation should be separate from datasets
    # TODO: disent/data should be datasets, and disent/datasets should be samplers that wrap disent/data

    def __init__(self, base_data: GroundTruthData, transform=None):
        self.base_data = base_data
        super().__init__(transform=transform)

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self.base_data.factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self.base_data.factor_sizes

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self.base_data.observation_shape

    def _get_observation(self, idx):
        return self.base_data[idx]


class AdversarialOptimizedData(TransformDataset):

    def __init__(self, h5_path: str, base_data: GroundTruthData, transform=None):
        # normalize hd5f data
        def _normalize_hdf5(x):
            c, h, w = x.shape
            if c in (1, 3):
                return np.moveaxis(x, 0, -1)
            return x
        # get the data
        self.hdf5_data = Hdf5Dataset(h5_path, transform=_normalize_hdf5)
        # checks
        assert isinstance(base_data, GroundTruthData), f'base_data must be an instance of {repr(GroundTruthData.__name__)}, got: {repr(base_data)}'
        assert len(base_data) == len(self.hdf5_data), f'length of base_data: {len(base_data)} does not match length of hd5f data: {len(self.hdf5_data)}'
        # initialize
        super().__init__(base_data=base_data, transform=transform)

    def _get_observation(self, idx):
        return self.hdf5_data[idx]
