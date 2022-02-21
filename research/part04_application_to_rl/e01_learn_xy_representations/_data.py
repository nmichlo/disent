#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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
from typing import Optional
from typing import Tuple

import numpy as np

from disent.dataset.data import GroundTruthData


log = logging.getLogger(__name__)


# ========================================================================= #
# Dataset                                                                   #
# ========================================================================= #


class XYSingleSquareData(GroundTruthData):

    name = 'xy_single_square'

    factor_names = ('x', 'y')

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return (self._placements, self._placements)

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, 1

    def __init__(
        self,
        square_size: int = 8,                # square width and height
        image_size: int = 64,                # image width and height
        grid_size: Optional[int] = None,     # limit the number of square placements along an axis, automatically set as the maximum valid
        grid_spacing: Optional[int] = None,  # how far apart the square is spaced, buy default this is the square size, meaning no overlap!
        transform=None,
    ):
        if grid_spacing is None:
            grid_spacing = square_size
        # vars
        self._width = image_size         # image width and height
        self._square_size = square_size  # square width and height
        self._spacing = grid_spacing     # spacing between square positions
        self._placements = (self._width - self._square_size) // grid_spacing + 1  # number of positions the square can be in along an axis
        # maximum placements
        if grid_size is not None:
            if (grid_size > self._placements):
                log.warning(f'number of possible placements: {self._placements} is less than the given grid size: {grid_size}, reduced grid size from: {grid_size} -> {self._placements}')
            self._placements = min(self._placements, grid_size)
        # center elements
        self._offset = (self._width - (self._square_size + (self._placements-1)*self._spacing)) // 2
        # initialise parents -- they depend on self.factors
        super().__init__(transform=transform)

    def _get_observation(self, idx):
        # get factors == grid position/index
        fx, fy = self.idx_to_pos(idx)
        offset, space, size = self._offset, self._spacing, self._square_size
        # draw square onto image
        obs = np.zeros(self.img_shape, dtype='uint8')
        x, y = offset + space * fx, offset + space * fy
        obs[y:y+size, x:x+size, :] = 255
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
