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

import numpy as np

from disent.dataset.data._groundtruth import GroundTruthData

log = logging.getLogger(__name__)


# ========================================================================= #
# xy squares data                                                           #
# ========================================================================= #


class XYBlocksData(GroundTruthData):

    """
    Dataset that generates all possible permutations of xor'd squares of
    different scales moving across the grid.

    This dataset is designed not to overlap in the reconstruction loss space, but xor'ing may be too
    complex to learn efficiently, and some sizes of factors may be too small (eg. biggest
    square moving only has two positions)
    """

    COLOR_PALETTES_1 = {
        "white": [
            [255],
        ],
        "greys_halves": [
            [128],
            [255],
        ],
        "greys_quarters": [
            [64],
            [128],
            [192],
            [255],
        ],
        # alias for white, so that we can just set `rgb=False`
        "rgb": [
            [255],
        ],
    }

    COLOR_PALETTES_3 = {
        "white": [
            [255, 255, 255],
        ],
        # THIS IS IDEAL.
        "rgb": [
            [255, 000, 000],
            [000, 255, 000],
            [000, 000, 255],
        ],
        "colors": [
            [255, 000, 000],
            [000, 255, 000],
            [000, 000, 255],
            [255, 255, 000],
            [000, 255, 255],
            [255, 000, 255],
            [255, 255, 255],
        ],
    }

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self._factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._factor_sizes

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self._img_shape

    def __init__(
        self,
        grid_size: int = 64,
        grid_levels: Tuple[int, ...] = (1, 2, 3),
        rgb: bool = True,
        palette: str = "rgb",
        invert_bg: bool = False,
        transform=None,
    ):
        # colors
        self._rgb = rgb
        if palette != "rgb":
            log.warning("rgb palette is not being used, might overlap for the reconstruction loss.")
        if rgb:
            assert (
                palette in XYBlocksData.COLOR_PALETTES_3
            ), f"{palette=} must be one of {list(XYBlocksData.COLOR_PALETTES_3.keys())}"
            self._colors = np.array(XYBlocksData.COLOR_PALETTES_3[palette])
        else:
            assert (
                palette in XYBlocksData.COLOR_PALETTES_1
            ), f"{palette=} must be one of {list(XYBlocksData.COLOR_PALETTES_1.keys())}"
            self._colors = np.array(XYBlocksData.COLOR_PALETTES_1[palette])

        # bg colors
        self._bg_color = 255 if invert_bg else 0  # we dont need rgb for this
        assert not np.any(
            [np.all(self._bg_color == color) for color in self._colors]
        ), f"Color conflict with background: {self._bg_color} ({invert_bg=}) in {self._colors}"

        # grid
        grid_levels = np.arange(1, grid_levels + 1) if isinstance(grid_levels, int) else np.array(grid_levels)
        assert np.all(grid_size % (2**grid_levels) == 0), f"{grid_size=} is not divisible by pow(2, {grid_levels=})"
        assert np.all(grid_levels[:-1] <= grid_levels[1:])
        self._grid_size = grid_size
        self._grid_levels = grid_levels
        self._grid_dims = len(grid_levels)

        # axis sizes
        self._axis_divisions = 2**self._grid_levels
        assert (
            len(self._axis_divisions) == self._grid_dims and np.all(grid_size % self._axis_divisions) == 0
        ), "This should never happen"
        self._axis_division_sizes = grid_size // self._axis_divisions

        # info
        self._factor_names = tuple([f"{prefix}-{d}" for prefix in ["color", "x", "y"] for d in self._axis_divisions])
        self._factor_sizes = tuple([len(self._colors)] * self._grid_dims + list(self._axis_divisions) * 2)
        self._img_shape = (grid_size, grid_size, 3 if self._rgb else 1)

        # initialise
        super().__init__(transform=transform)

    def _get_observation(self, idx):
        positions = self.idx_to_pos(idx)
        cs, xs, ys = (
            positions[: self._grid_dims * 1],
            positions[self._grid_dims * 1 : self._grid_dims * 2],
            positions[self._grid_dims * 2 :],
        )
        assert len(xs) == len(ys) == len(cs)
        # GENERATE
        obs = np.full(self.img_shape, self._bg_color, dtype=np.uint8)
        for i, (x, y, s, c) in enumerate(zip(xs, ys, self._axis_division_sizes, cs)):
            obs[y * s : (y + 1) * s, x * s : (x + 1) * s, :] = (
                self._colors[c] if np.any(obs[y * s, x * s, :] != self._colors[c]) else self._bg_color
            )
        # RETURN
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
