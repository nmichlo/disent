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

import warnings
from typing import Optional
from typing import Tuple

import numpy as np

from disent.dataset.data._groundtruth import GroundTruthData


# ========================================================================= #
# helper                                                                    #
# ========================================================================= #


_R, _G, _B, _Y, _C, _M, _W = np.array([
    [255, 000, 000], [000, 255, 000], [000, 000, 255],  # R, G, B
    [255, 255, 000], [000, 255, 255], [255, 000, 255],  # Y, C, M
    [255, 255, 255],                                    # white
])


def _shades(num: int, shades):
    all_shades = np.array([shade * i // num for i in range(1, num+1) for shade in np.array(shades)])
    assert all_shades.dtype in ('int64', 'int32')
    return all_shades


# ========================================================================= #
# xy object data                                                            #
# ========================================================================= #


class XYObjectData(GroundTruthData):

    """
    Dataset that generates all possible permutations of a square placed on a square grid,
    with varying scale and colour

    *NB* for most of these color palettes, there should be
         an extra ground truth factor that represents shade.
         We purposely leave this out to hinder disentanglement! It is subjective!
    """

    name = 'xy_object'

    COLOR_PALETTES_1 = {
        'greys_1':   _shades(1, [[255]]),
        'greys_2':   _shades(2, [[255]]),
        'greys_4':   _shades(4, [[255]]),
        # aliases for greys so that we can just set `rgb=False` and it still works
        'rainbow_1': _shades(1, [[255]]),
        'rainbow_2': _shades(2, [[255]]),
        'rainbow_4': _shades(4, [[255]]),
    }

    COLOR_PALETTES_3 = {
        # grey
        'greys_1': _shades(1, [_W]),
        'greys_2': _shades(2, [_W]),
        'greys_4': _shades(4, [_W]),
        # colors -- white here and the incorrect ordering may throw off learning ground truth factors
        'colors_1': _shades(1, [_R, _G, _B, _Y, _C, _M, _W]),
        'colors_2': _shades(2, [_R, _G, _B, _Y, _C, _M, _W]),
        'colors_4': _shades(4, [_R, _G, _B, _Y, _C, _M, _W]),
        # rgb
        'rgb_1': _shades(1, [_R, _G, _B]),
        'rgb_2': _shades(2, [_R, _G, _B]),
        'rgb_4': _shades(4, [_R, _G, _B]),
        # rainbows -- these colors are mostly ordered correctly to align with gt factors
        'rainbow_1': _shades(1, [_R, _Y, _G, _C, _B, _M]),
        'rainbow_2': _shades(2, [_R, _Y, _G, _C, _B, _M]),
        'rainbow_4': _shades(4, [_R, _Y, _G, _C, _B, _M]),
    }

    factor_names = ('x', 'y', 'scale', 'color')

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._placements, self._placements, len(self._square_scales), len(self._colors)

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, (3 if self._rgb else 1)

    def __init__(
        self,
        grid_size: int = 64,
        grid_spacing: int = 2,
        min_square_size: int = 7,
        max_square_size: int = 15,
        square_size_spacing: int = 2,
        rgb: bool = True,
        palette: str = 'rainbow_4',
        transform=None,
    ):
        # generation
        self._rgb = rgb
        # check the pallete name
        assert len(str.split(palette, '_')) == 2, f'palette name must follow format: `<palette-name>_<brightness-levels>`, got: {repr(palette)}'
        # get the color palette
        color_palettes = (XYObjectData.COLOR_PALETTES_3 if rgb else XYObjectData.COLOR_PALETTES_1)
        if palette not in color_palettes:
            raise KeyError(f'color palette: {repr(palette)} does not exist for rgb={repr(rgb)}, select one of: {sorted(color_palettes.keys())}')
        self._colors = color_palettes[palette]
        assert self._colors.ndim == 2
        assert self._colors.shape[-1] == (3 if rgb else 1)
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
        obs = np.zeros(self.img_shape, dtype=np.uint8)
        obs[y:y+s, x:x+s] = self._colors[c]
        return obs


class XYOldObjectData(XYObjectData):

    name = 'xy_object_shaded'

    def __init__(self, grid_size=64, grid_spacing=1, min_square_size=3, max_square_size=9, square_size_spacing=2, rgb=True, palette='colors', transform=None):
        super().__init__(
            grid_size=grid_size,
            grid_spacing=grid_spacing,
            min_square_size=min_square_size,
            max_square_size=max_square_size,
            square_size_spacing=square_size_spacing,
            rgb=rgb,
            palette=palette,
            transform=transform,
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


class XYObjectShadedData(XYObjectData):
    """
    Dataset that generates all possible permutations of a square placed on a square grid,
    with varying scale and colour

    - This is like `XYObjectData` but has an extra factor that represents the shade.
    """

    factor_names = ('x', 'y', 'scale', 'intensity', 'color')

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._placements, self._placements, len(self._square_scales), self._brightness_levels, len(self._colors)

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, (3 if self._rgb else 1)

    def __init__(
        self,
        grid_size: int = 64,
        grid_spacing: int = 2,
        min_square_size: int = 7,
        max_square_size: int = 15,
        square_size_spacing: int = 2,
        rgb: bool = True,
        palette: str = 'rainbow_4',
        brightness_levels: Optional[int] = None,
        transform=None,
    ):
        parts = palette.split('_')
        if len(parts) > 1:
            # extract num levels from the string
            palette, b_levels = parts
            b_levels = int(b_levels)
            # handle conflict between brightness_levels and palette
            if brightness_levels is None:
                brightness_levels = b_levels
            else:
                warnings.warn(f'palette ends with brightness_levels integer: {repr(b_levels)} (ignoring) but actual brightness_levels parameter was already specified: {repr(brightness_levels)} (using)')
        # check the brightness_levels
        assert isinstance(brightness_levels, int), f'brightness_levels must be an integer, got: {type(brightness_levels)}'
        assert 1 <= brightness_levels, f'brightness_levels must be >= 1, got: {repr(brightness_levels)}'
        self._brightness_levels = brightness_levels
        # initialize parent
        super().__init__(
            grid_size=grid_size,
            grid_spacing=grid_spacing,
            min_square_size=min_square_size,
            max_square_size=max_square_size,
            square_size_spacing=square_size_spacing,
            rgb=rgb,
            palette=f'{palette}_1',
            transform=transform,
        )

    def _get_observation(self, idx):
        x, y, s, b, c = self.idx_to_pos(idx)
        s = self._square_scales[s]
        r = (self._max_square_size - s) // 2
        x, y = self._spacing*x + r, self._spacing*y + r
        # GENERATE
        obs = np.zeros(self.img_shape, dtype=np.uint8)
        obs[y:y+s, x:x+s] = self._colors[c] * (b + 1) // self._brightness_levels
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
