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

from disent.dataset.data._groundtruth import GroundTruthData


# ========================================================================= #
# xy grid data                                                           #
# ========================================================================= #


class XYObjectData(GroundTruthData):

    """
    Dataset that generates all possible permutations of a square placed on a square grid,
    with varying scale and colour

    - Does not seem to learn with a VAE when square size is equal to 1
      (This property may be explained in the paper "Understanding disentanglement in Beta-VAEs")
    """

    COLOR_PALETTES_1 = {
        'white': [
            [255],
        ],
        'greys_halves': [
            [128],
            [255],
        ],
        'greys_quarters': [
            [64],
            [128],
            [192],
            [255],
        ],
        'colors': [
            [64],
            [128],
            [192],
            [255],
        ],
    }

    COLOR_PALETTES_3 = {
        'white': [
            [255, 255, 255],
        ],
        'greys_halves': [
            [128, 128, 128],
            [255, 255, 255],
        ],
        'greys_quarters': [
            [64, 64, 64],
            [128, 128, 128],
            [192, 192, 192],
            [255, 255, 255],
        ],
        'rgb': [
            [255, 000, 000],
            [000, 255, 000],
            [000, 000, 255],
        ],
        'colors': [
            [255, 000, 000], [000, 255, 000], [000, 000, 255],
            [255, 255, 000], [000, 255, 255], [255, 000, 255],
            [255, 255, 255],
        ],
        'colors_halves': [
            [128, 000, 000], [000, 128, 000], [000, 000, 128],
            [128, 128, 000], [000, 128, 128], [128, 000, 128],
            [128, 128, 128],
            [255, 000, 000], [000, 255, 000], [000, 000, 255],
            [255, 255, 000], [000, 255, 255], [255, 000, 255],
            [255, 255, 255],
        ],
    }

    factor_names = ('x', 'y', 'scale', 'color')

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._placements, self._placements, len(self._square_scales), len(self._colors)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, (3 if self._rgb else 1)

    def __init__(self, grid_size=64, grid_spacing=1, min_square_size=3, max_square_size=9, square_size_spacing=2, rgb=True, palette='colors', transform=None):
        # generation
        self._rgb = rgb
        if rgb:
            self._colors = np.array(XYObjectData.COLOR_PALETTES_3[palette])
        else:
            self._colors = np.array(XYObjectData.COLOR_PALETTES_1[palette])
        # image sizes
        self._width = grid_size
        # square scales
        assert min_square_size <= max_square_size
        self._max_square_size, self._max_square_size = min_square_size, max_square_size
        self._square_scales = np.arange(min_square_size, max_square_size+1, square_size_spacing)
        # x, y
        self._spacing = grid_spacing
        self._placements = (self._width - max_square_size) // grid_spacing + 1
        super().__init__(transform=transform)
    
    def _get_observation(self, idx):
        x, y, s, c = self.idx_to_pos(idx)
        s = self._square_scales[s]
        r = (self._max_square_size - s) // 2
        x, y = self._spacing*x + r, self._spacing*y + r
        # GENERATE
        obs = np.zeros(self.observation_shape, dtype=np.uint8)
        obs[y:y+s, x:x+s] = self._colors[c]
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
