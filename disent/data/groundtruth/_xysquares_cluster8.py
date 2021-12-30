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

from disent.util import iter_chunks

log = logging.getLogger(__name__)


# ========================================================================= #
# xy multi grid data                                                        #
# ========================================================================= #


class XYSquaresClusterData4(GroundTruthData):
    """
    Dataset that generates all possible permutations of 3 (R, G, B) coloured
    squares placed on a square grid.

    This dataset is designed to not overlap in the reconstruction loss space.
    (if the spacing is set correctly.)
    """

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return ('x_R', 'y_R', 'x_G', 'y_G', 'x_B', 'y_B')[:self._num_squares * 2]

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return (self._placements, self._placements) * self._num_squares  # R, G, B squares

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, (3 if self._rgb else 1)


    def __init__(self, square_size=8, grid_size=64, grid_spacing=None, num_squares=1,num_clusters=8, rgb=False, no_warnings=False,
                 fill_value=None, max_placements=None, outlines=False):
        if grid_spacing is None:
            grid_spacing = square_size
        if grid_spacing < square_size and not no_warnings:
            log.warning(f'overlap between squares for reconstruction loss, {grid_spacing} < {square_size}')
        # color
        self._rgb = rgb
        self._fill_value = fill_value if (fill_value is not None) else 255
        assert isinstance(self._fill_value, int)
        assert 0 < self._fill_value <= 255, f'0 < {self._fill_value} <= 255'
        # image sizes
        self._width = grid_size
        # number of squares
        self._num_squares = num_squares
        self.num_clusters = num_clusters
        assert 1 <= num_squares <= 3, 'Only 1, 2 or 3 squares are supported!'
        # square scales
        self._square_size = square_size
        # x, y
        self._spacing = grid_spacing
        self._placements = (self._width - self._square_size) // grid_spacing + 1
        # maximum placements
        if max_placements is not None:
            assert isinstance(max_placements, int)
            assert max_placements > 0
            self._placements = min(self._placements, max_placements)
        # outline regions
        self.outlines = outlines
        # center elements
        self._offset = (self._width - (self._square_size + (self._placements - 1) * self._spacing)) // 2
        super().__init__()

    def __getitem__(self, idx):
        # get factors
        factors = self.idx_to_pos(idx)
        allowed_factors = self.allowed_factor(factors,num_clusters=self.num_clusters )
        #print(factors,':', allowed_factors)
        offset, space, size = self._offset, self._spacing, self._square_size
        # GENERATE
        obs = np.zeros(self.observation_shape, dtype=np.uint8)
        if self.outlines:
            obs[self._width//2] = self._fill_value
            obs[:, self._width//2] = self._fill_value

        for i, (fx, fy) in enumerate(iter_chunks(allowed_factors, 2)):

            #(Fx, Fy) = self.make_base_factor(fx, fy)

            x, y = offset + space * fx, offset + space * fy

            if self._rgb:
                obs[y:y + size, x:x + size, i] = self._fill_value
            else:
                obs[y:y + size, x:x + size, :] = self._fill_value
        return obs

'''
import logging
from typing import Tuple
from disent.data.groundtruth.base import GroundTruthData
import numpy as np

from disent.util import iter_chunks

log = logging.getLogger(__name__)


# ======================================================================== #
# xy multi grid data                                                        #
# ========================================================================= #


class XYSquaresData(GroundTruthData):
    """
    Dataset that generates all possible permutations of 3 (R, G, B) coloured
    squares placed on a square grid.

    This dataset is designed to not overlap in the reconstruction loss space.
    (if the spacing is set correctly.)

    X,Y sampling regions are created to align with the number of modes selected.

    """

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return ('x_R', 'y_R', 'x_G', 'y_G', 'x_B', 'y_B')[:self._num_squares * 2]

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return (self._placements // 2, self._placements / 2) * self._num_squares  # R, G, B squares

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, (3 if self._rgb else 1)

    def __init__(self, square_size=8, grid_size=64, grid_spacing=None, num_squares=3, num_modes=3, rgb=True,
                 no_warnings=False, fill_value=None, max_placements=None):
        if grid_spacing is None:
            grid_spacing = square_size
        if grid_spacing < square_size and not no_warnings:
            log.warning(f'overlap between squares for reconstruction loss, {grid_spacing} < {square_size}')
        self._num_modes = num_modes
        # color
        self._rgb = rgb
        self._fill_value = fill_value if (fill_value is not None) else 255
        assert isinstance(self._fill_value, int)
        assert 0 < self._fill_value <= 255, f'0 < {self._fill_value} <= 255'
        # image sizes
        self._width = grid_size
        # number of squares
        self._num_squares = num_squares
        assert 1 <= num_squares <= 3, 'Only 1, 2 or 3 squares are supported!'
        # square scales
        self._square_size = square_size
        # x, y
        self._spacing = grid_spacing
        self._placements = ((self._width - self._square_size) // self._spacing + 1)
        # maximum placements (limit on placements)

        if max_placements is not None:
            assert isinstance(max_placements, int)
            self._placements = min(self._placements, max_placements)
        # center elements
        self._offset = (self._width - (self._square_size + (self._placements - 1) * self._spacing)) // 2
        self.factors = []
        for x in range(0, 8):
            for y in range(0, 8):
                self.factors.append([x, y])
        super().__init__()

    def __getitem__(self, idx):

        # GENERATE
        factors = self.factors[idx]
        fx, fy = factors
        offset, space, size = self._offset, self._spacing, self._square_size
        # GENERATE
        obs = np.zeros(self.observation_shape, dtype=np.uint8)

        x, y = offset + space * fx, offset + space * fy
        print('x: ', x, 'y: ', y)
        obs[y:y + size, x:x + size, :] = self._fill_value
        return obs

    def __getitem__(self, idx):
        # get factors
        factors = self.idx_to_pos(idx)
        print(factors)
        offset, space, size = self._offset, self._spacing, self._square_size
        # GENERATE
        obs = np.zeros(self.observation_shape, dtype=np.uint8)

        for i, (fx, fy) in enumerate(iter_chunks(factors, 2)):
            x, y = offset + space * fx, offset + space * fy
            if self._rgb:
                obs[y:y + size, x:x + size, i] = self._fill_value
            else:
                obs[y:y + size, x:x + size, :] = self._fill_value
        return obs





        # carving up space to easily see what's going on visually
        x_bound = np.int(self._width)
        y_bound = np.int(self._width)

        #self._blank_obs_sum = np.sum(obs)/4 + self._square_size*self._square_size

        for i, (fx, fy) in enumerate(iter_chunks(factors, 2)):
            x, y = offset + space * fx, offset + space * fy
            # quadrant 2 to quadrant 1
            # obs[:x_bound, x_bound:, :]
            if x <= (x_bound - self._square_size) and (y <= (x_bound - self._square_size)):
                obs = np.zeros(self.observation_shape, dtype=np.uint8)
                obs[x_bound] = self._fill_value
                obs[:, x_bound] = self._fill_value
                obs[y:y + size, x:x + size, :] = self._fill_value
                return obs

            elif x > (x_bound - self._square_size) and (y > (x_bound - self._square_size)):
                obs = np.zeros(self.observation_shape, dtype=np.uint8)
                obs[x_bound] = self._fill_value
                obs[:, x_bound] = self._fill_value
                obs[y:y + size, x:x + size, :] = self._fill_value
                return obs
            else:
                continue
                #x = x_bound + (x - x_bound) - self._square_size

            # quadrant 3 to quadrant 4
            #obs[x_bound:, :x_bound, :]
            #if (x < x_bound) and (y > x_bound):
                #continue
                #y = x_bound + (x_bound-y) - self._square_size

            #if self._rgb:
                #obs[y:y+size, x:x+size, i] = self._fill_value
            #else:
                #obs[y:y + size, x:x + size, :] = self._fill_value

                # quadrant 1
                #if obs[:x_bound, :x_bound, :].sum != 0:

                    #obs = np.rot90(np.fliplr(obs))

                # quadrant 2
                #if np.sum(obs[:x_bound, x_bound:, :]) > self._blank_obs_sum:
                    #print(np.sum(obs[:x_bound, x_bound:, :]))

                    #obs[:x_bound, x_bound:, :] = self._fill_value


                # quadrant 3
                #if np.sum(obs[x_bound:, :x_bound, :]) > self._blank_obs_sum:
                    #obs[x_bound:, :x_bound, :] = self._fill_value

                # quadrant 4
                # obs[x_bound:, x_bound:, :] = self._fill_value


        # limit to spatial mode region
        #if x >= x_bound:
            #if y <= y_bound:

                #x = np.abs(x - x_bound)
                #obs = np.zeros(self.observation_shape, dtype=np.uint8)

        #if y >= y_bound:
            #if x <= x_bound:

                #y = np.abs(y - y_bound)
                #obs = np.zeros(self.observation_shape, dtype=np.uint8)

        #return obs
'''

# ======================================================================= #
# END                                                                       #
# ========================================================================= #
