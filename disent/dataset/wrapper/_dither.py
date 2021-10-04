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
import numpy as np
from torch.utils.data import Dataset

from disent.dataset.data import GroundTruthData
from disent.dataset.util.state_space import StateSpace
from disent.dataset.wrapper._base import WrappedDataset
from disent.util.math.dither import nd_dither_matrix


log = logging.getLogger(__name__)


# ========================================================================= #
# Dithered Dataset                                                          #
# ========================================================================= #


class DitheredDataset(WrappedDataset):

    def __init__(self, gt_data: GroundTruthData, dither_n: int = 2, keep_ratio: float = 1):
        assert 0 < keep_ratio <= 1.0
        assert isinstance(gt_data, GroundTruthData)
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        self._gt_data = gt_data
        # data space
        data_idx = np.arange(len(gt_data))
        data_pos = gt_data.idx_to_pos(data_idx)
        # dmat space
        d_mat = nd_dither_matrix(n=dither_n, d=self._gt_data.num_factors, norm=True) < keep_ratio
        d_states = StateSpace(d_mat.shape)
        # data space to dmat space
        dmat_pos = data_pos % dither_n
        dmat_idx = d_states.pos_to_idx(dmat_pos)
        mask = d_mat.flatten()[dmat_idx]
        # convert mask to indices
        self._indices = np.arange(len(gt_data))[mask]
        # -~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
        assert len(self._indices) > 0
        # check values & count compared to ratio
        unique_count_map = {False: 0, True: 0, **{u: c for u, c in zip(*np.unique(mask, return_counts=True))}}
        assert len(unique_count_map) == 2
        assert unique_count_map[True] > 0
        assert sum(unique_count_map.values()) == len(gt_data)
        log.info(f'[n={dither_n}] keep ratio: {keep_ratio:.2f} actual ratio: {unique_count_map[True] / sum(unique_count_map.values()):.2f}')

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, item):
        return self._gt_data[self._indices[item]]

    @property
    def data(self) -> Dataset:
        return self._gt_data


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
