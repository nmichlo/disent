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

from disent.dataset.data._groundtruth__xysquares import XYSquaresData


# ========================================================================= #
# xy multi grid data                                                        #
# ========================================================================= #


class XColumnsData(XYSquaresData):

    """
    Michlo et al.
    https://github.com/nmichlo/msc-research

    Like XYSquares, but has a single column that moves left and right, instead of across a grid.
    - This dataset is also adversarial!
    """

    name = 'x_columns'

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return ('x_R', 'x_G', 'x_B')[:self._num_squares]

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return (self._placements,) * self._num_squares

    def _get_observation(self, idx):
        # get factors
        factors = self.idx_to_pos(idx)
        offset, space, size = self._offset, self._spacing, self._square_size
        # GENERATE
        obs = np.zeros(self.img_shape, dtype=self._dtype)
        for i, fx in enumerate(factors):
            x = offset + space * fx
            if self._rgb:
                obs[:, x:x+size, i] = self._fill_value
            else:
                obs[:, x:x+size, :] = self._fill_value
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
