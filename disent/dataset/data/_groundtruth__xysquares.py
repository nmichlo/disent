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
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from disent.dataset.data._groundtruth import GroundTruthData
from disent.util.iters import iter_chunks


log = logging.getLogger(__name__)


# ========================================================================= #
# xy multi grid data                                                        #
# ========================================================================= #


class XYSquaresMinimalData(GroundTruthData):
    """
    Michlo et al.
    https://github.com/nmichlo/msc-research

    Dataset that generates all possible permutations of 3 (R, G, B) coloured
    squares placed on a square grid. This dataset is designed to not overlap
    in the reconstruction loss space.

    If you use this in your work, please cite: https://github.com/nmichlo/disent

    NOTE: Unlike XYSquaresData, XYSquaresMinimalData is the bare-minimum class
          to generate the same results as the default values for XYSquaresData,
          this class is a fair bit faster (~0.8x)!
          - All 3 squares are returned, in RGB, each square is size 8, with
            non-overlapping grid spacing set to 8 pixels, in total leaving
            8*8*8*8*8*8 factors. Images are uint8 with fill values 0 (bg)
            and 255 (fg).

    This dataset is adversarial in nature to auto-encoders that use pixel-wise reconstruction losses.
    - The AE or VAE cannot order observations in the latent space. Disentanglement performance suffers hugely.
    - This is because this dataset has constant distance between data-points as measured by the reconstruction loss.
      If the grid spacing is lowered, such that overlap is introduced, then disentanglement can occur again!
    """

    name = 'xy_squares_minimal'

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return 'x_R', 'y_R', 'x_G', 'y_G', 'x_B', 'y_B'

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return 8, 8, 8, 8, 8, 8  # R, G, B squares

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return 64, 64, 3

    def _get_observation(self, idx):
        # get factors
        factors = np.reshape(np.unravel_index(idx, self.factor_sizes), (-1, 2))
        # GENERATE
        obs = np.zeros(self.img_shape, dtype=np.uint8)
        for i, (fx, fy) in enumerate(factors):
            x, y = 8 * fx, 8 * fy
            obs[y:y+8, x:x+8, i] = 255
        return obs


# ========================================================================= #
# xy multi grid data                                                        #
# ========================================================================= #


class XYSquaresData(GroundTruthData):
    """
    Michlo et al.
    https://github.com/nmichlo/msc-research

    Dataset that generates all possible permutations of 3 (R, G, B) coloured
    squares placed on a square grid. This dataset is designed to not overlap
    in the reconstruction loss space. (if the spacing is set correctly.)

    If you use this in your work, please cite: https://github.com/nmichlo/disent

    NOTE: Unlike XYSquaresMinimalData, XYSquaresData allows adjusting various aspects
          of the data that is generated, but the generation process is slower (~1.25x).

    This dataset is adversarial in nature to auto-encoders that use pixel-wise reconstruction losses.
    - The AE or VAE cannot order observations in the latent space. Disentanglement performance suffers hugely.
    - This is because this dataset has constant distance between data-points as measured by the reconstruction loss.
      If the grid spacing is lowered, such that overlap is introduced, then disentanglement can occur again!
    """

    name = 'xy_squares'

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return ('x_R', 'y_R', 'x_G', 'y_G', 'x_B', 'y_B')[:self._num_squares*2]

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return (self._placements, self._placements) * self._num_squares  # R, G, B squares

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, (3 if self._rgb else 1)

    def __init__(
        self,
        square_size: int = 8,
        image_size: int = 64,
        grid_size: Optional[int] = None,
        grid_spacing: Optional[int] = None,
        num_squares: int = 3,
        rgb: bool = True,
        fill_value: Optional[Union[float, int]] = None,
        dtype: Union[np.dtype, str] = np.uint8,
        no_warnings: bool = False,
        transform=None,
    ):
        """
        :param square_size: the size of the individual squares in pixels
        :param image_size: the image size in pixels
        :param grid_spacing: the step size between square positions on the grid. By
               default this is set to square_size which results in non-overlapping
               data if `grid_spacing >= square_size` Reducing this value such that
               `grid_spacing < square_size` results in overlapping data.
        :param num_squares: The number of squares drawn. `1 <= num_squares <= 3`
        :param rgb: Image has 3 channels if True, otherwise it is greyscale with 1 channel.
        :param no_warnings: If warnings should be disabled if overlapping.
        :param fill_value: The foreground value to use for filling squares, the default background value is 0.
        :param grid_size: The number of grid positions available for the square to be placed in. The square is centered if this is less than
        :param dtype: 
        """
        if grid_spacing is None:
            grid_spacing = square_size
        if (grid_spacing < square_size) and not no_warnings:
            log.warning(f'overlap between squares for reconstruction loss, {grid_spacing} < {square_size}')
        # color
        self._rgb = rgb
        self._dtype = np.dtype(dtype)
        # check fill values
        if self._dtype.kind == 'u':
            self._fill_value = 255 if (fill_value is None) else fill_value
            assert isinstance(self._fill_value, int)
            assert 0 < self._fill_value <= 255, f'0 < {self._fill_value} <= 255'
        elif self._dtype.kind == 'f':
            self._fill_value = 1.0 if (fill_value is None) else fill_value
            assert isinstance(self._fill_value, (int, float))
            assert 0.0 < self._fill_value <= 1.0, f'0.0 < {self._fill_value} <= 1.0'
        else:
            raise TypeError(f'invalid dtype: {self._dtype}, must be float or unsigned integer')
        # image sizes
        self._width = image_size
        # number of squares
        self._num_squares = num_squares
        assert 1 <= num_squares <= 3, 'Only 1, 2 or 3 squares are supported!'
        # square scales
        self._square_size = square_size
        # x, y
        self._spacing = grid_spacing
        self._placements = (self._width - self._square_size) // grid_spacing + 1
        # maximum placements
        if grid_size is not None:
            assert isinstance(grid_size, int)
            assert grid_size > 0
            if (grid_size > self._placements) and not no_warnings:
                log.warning(f'number of possible placements: {self._placements} is less than the given grid size: {grid_size}, reduced grid size from: {grid_size} -> {self._placements}')
            self._placements = min(self._placements, grid_size)
        # center elements
        self._offset = (self._width - (self._square_size + (self._placements-1)*self._spacing)) // 2
        # initialise parents -- they depend on self.factors
        super().__init__(transform=transform)

    def _get_observation(self, idx):
        # get factors
        factors = self.idx_to_pos(idx)
        offset, space, size = self._offset, self._spacing, self._square_size
        # GENERATE
        obs = np.zeros(self.img_shape, dtype=self._dtype)
        for i, (fx, fy) in enumerate(iter_chunks(factors, 2)):
            x, y = offset + space * fx, offset + space * fy
            if self._rgb:
                obs[y:y+size, x:x+size, i] = self._fill_value
            else:
                obs[y:y+size, x:x+size, :] = self._fill_value
        return obs


# ========================================================================= #
# xy minimal single square dataset                                          #
# ========================================================================= #


class XYSingleSquareData(GroundTruthData):
    """
    Michlo et al.
    https://github.com/nmichlo/msc-research

    This is a very basic version of XYSquares that only has a
    single object that moves around, instead of 3.
    - This dataset is still adversarial if the grid_spacing is not adjusted.
    """

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
        no_warnings: bool = False,
        transform=None,
    ):
        if grid_spacing is None:
            grid_spacing = square_size
        if (grid_spacing < square_size) and not no_warnings:
            log.warning(f'overlap between squares for reconstruction loss, {grid_spacing} < {square_size}')
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
