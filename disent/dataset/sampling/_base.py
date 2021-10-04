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
from typing import Tuple


# ========================================================================= #
# Base Sampler                                                              #
# ========================================================================= #


class BaseDisentSampler(object):

    def uninit_copy(self) -> 'BaseDisentSampler':
        raise NotImplementedError

    def __init__(self, num_samples: int):
        self._num_samples = num_samples

    @property
    def num_samples(self) -> int:
        return self._num_samples

    __initialized = False

    @final
    def init(self, dataset) -> 'BaseDisentSampler':
        if self.__initialized:
            raise RuntimeError(f'Sampler: {repr(self.__class__.__name__)} has already been initialized, are you sure it is not being reused?')
        # initialize
        self.__initialized = True
        self._init(dataset)
        return self

    def _init(self, dataset):
        """
        you can override this method to initialize the class!
        """
        pass

    @property
    def is_init(self) -> bool:
        return self.__initialized

    def _sample_idx(self, idx: int) -> Tuple[int, ...]:
        raise NotImplementedError

    def __call__(self, idx: int) -> Tuple[int, ...]:
        # check that we have been initialized!
        if not self.is_init:
            raise RuntimeError(f'{self.__class__.__name__} has not been initialized! call `sampler.init(gt_data)`')
        # sample values
        idxs = self._sample_idx(idx)
        # check values
        if len(idxs) != self.num_samples:
            raise RuntimeError(f'{self.__class__.__name__} returned incorrect number of samples, required: {self.num_samples}, got: {len(idxs)}')
        # return values
        return idxs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
