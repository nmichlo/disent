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

import logging
from typing import Union

import numpy as np
import torch
from torch.utils.data import Dataset

from disent.dataset.data import GroundTruthData
from disent.dataset.wrapper._base import WrappedDataset
from disent.util.math.random import random_choice_prng


log = logging.getLogger(__name__)


# ========================================================================= #
# Masked Dataset                                                            #
# ========================================================================= #


DataTypeHint = Union[GroundTruthData, np.ndarray, torch.Tensor]
MaskTypeHint = Union[str, np.ndarray]


def load_mask_indices(length: int, mask_or_indices: MaskTypeHint):
    # load as numpy mask if it is a string!
    if isinstance(mask_or_indices, str):
        mask_or_indices = np.load(mask_or_indices)['mask']
    # load data
    assert isinstance(mask_or_indices, np.ndarray)
    # check inputs
    if mask_or_indices.dtype == 'bool':
        # boolean values
        assert length == len(mask_or_indices)
        indices = np.arange(length)[mask_or_indices]
    else:
        # integer values
        assert len(np.unique(mask_or_indices)) == len(mask_or_indices)
        assert np.min(mask_or_indices) >= 0
        assert np.max(mask_or_indices) < length
        indices = mask_or_indices
    # check that we have at least 1 value
    assert len(indices) > 0
    assert len(indices) <= length
    # return values
    return indices


class MaskedDataset(WrappedDataset):

    def __init__(self, data: DataTypeHint, mask: MaskTypeHint, randomize: bool = False):
        assert isinstance(data, (GroundTruthData, torch.Tensor, np.ndarray))
        n = len(data)
        # save values
        self._data = data
        self._indices = load_mask_indices(n, mask)
        # randomize
        if randomize:
            l = len(self._indices)
            self._indices = load_mask_indices(n, random_choice_prng(n, size=l, replace=False))
            assert len(self._indices) == l
            log.info(f'replaced mask: {l}/{n} ({l/n:.3f}) with randomized mask!')

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._data[self._indices[idx]]

    @property
    def data(self) -> Dataset:
        return self._data


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
