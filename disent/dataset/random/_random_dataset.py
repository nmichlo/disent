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

import numpy as np
from torch.utils.data import Dataset
from disent.dataset._augment_util import AugmentableDataset
from disent.util import LengthIter


# ========================================================================= #
# Randomly Paired Dataset                                                   #
# ========================================================================= #


class RandomDataset(Dataset, LengthIter, AugmentableDataset):

    def __init__(
            self,
            data: LengthIter,
            transform=None,
            augment=None,
            num_samples=1,
    ):
        self._data = data
        self._num_samples = num_samples
        # augmentable dataset
        self._transform = transform
        self._augment = augment

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Augmentable Dataset Overrides                                         #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def transform(self):
        return self._transform

    @property
    def augment(self):
        return self._augment

    def _get_augmentable_observation(self, idx):
        return self._data[idx]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        # sample indices
        indices = (idx, *np.random.randint(0, len(self), size=self._num_samples-1))
        # get data
        return self.dataset_get_observation(*indices)


# ========================================================================= #
# End                                                                       #
# ========================================================================= #

