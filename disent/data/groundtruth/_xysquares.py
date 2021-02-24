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
from typing import Tuple
from disent.data.groundtruth.base import GroundTruthData
import numpy as np
from disent.util import chunked

log = logging.getLogger(__name__)


# ========================================================================= #
# xy multi grid data                                                        #
# ========================================================================= #


class XYSquaresData(GroundTruthData):

    """
    Dataset that generates all possible permutations of 3 (R, G, B) coloured
    squares placed on a square grid.
    
    This dataset is designed to not overlap in the reconstruction loss space.
    (if the spacing is set correctly.)
    """

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return ('x_R', 'y_R', 'x_G', 'y_G', 'x_B', 'y_B')[:self._num_squares*2]

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return (self._placements, self._placements) * self._num_squares  # R, G, B squares

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, (3 if self._rgb else 1)

    def __init__(self, square_size=8, grid_size=64, grid_spacing=None, num_squares=3, rgb=True):
        if grid_spacing is None:
            grid_spacing = square_size
        if grid_spacing < square_size:
            log.warning(f'overlap between squares for reconstruction loss, {grid_spacing} < {square_size}')
        # color
        self._rgb = rgb
        # image sizes
        self._width = grid_size
        # number of squares
        self._num_squares = num_squares
        assert 1 <= num_squares <= 3, 'Only 1, 2 or 3 squares are supported!'
        # square scales
        self._square_size = square_size
        # x, y
        self._spacing = grid_spacing
        self._placements = (self._width - self._square_size) // grid_spacing + 1
        self._offset = (self._width - (self._placements * self._spacing)) // 2
        super().__init__()
    
    def __getitem__(self, idx):
        # get factors
        factors = self.idx_to_pos(idx)
        offset, space, size = self._offset, self._spacing, self._square_size
        # GENERATE
        obs = np.zeros(self.observation_shape, dtype=np.uint8)
        for i, (fx, fy) in enumerate(chunked(factors, 2)):
            x, y = offset + space * fx, offset + space * fy
            if self._rgb:
                obs[y:y+size, x:x+size, i] = 255
            else:
                obs[y:y+size, x:x+size, :] = 255
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
