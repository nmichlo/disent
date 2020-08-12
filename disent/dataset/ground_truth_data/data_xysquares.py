from typing import Tuple
from disent.dataset.ground_truth_data.base_data import GroundTruthData
import numpy as np


# ========================================================================= #
# xy grid data                                                           #
# ========================================================================= #


class XYSquaresData(GroundTruthData):

    """
    Dataset that generates all possible permutations of xor'd squares of
    different scales moving across the grid.
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
            [ 64],
            [128],
            [192],
            [255],
        ],
    }

    COLOR_PALETTES_3 = {
        'white': [
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
    }

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self._factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._factor_sizes

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._observation_shape
    
    def __init__(self, grid_size=64, grid_levels=(1, 2, 3), rgb=True, palette='rgb', invert_bg=False):
        # colors
        self._rgb = rgb
        if rgb:
            assert palette in XYSquaresData.COLOR_PALETTES_3, f'{palette=} must be one of {list(XYSquaresData.COLOR_PALETTES_3.keys())}'
            self._colors = np.array(XYSquaresData.COLOR_PALETTES_3[palette])
        else:
            assert palette in XYSquaresData.COLOR_PALETTES_1, f'{palette=} must be one of {list(XYSquaresData.COLOR_PALETTES_1.keys())}'
            self._colors = np.array(XYSquaresData.COLOR_PALETTES_1[palette])

        # bg colors
        self._bg_color = 255 if invert_bg else 0  # we dont need rgb for this
        assert not np.any([np.all(self._bg_color == color) for color in self._colors]), f'Color conflict with background: {self._bg_color} ({invert_bg=}) in {self._colors}'

        # grid
        grid_levels = np.arange(1, grid_levels+1) if isinstance(grid_levels, int) else np.array(grid_levels)
        assert np.all(grid_size % (2 ** grid_levels) == 0), f'{grid_size=} is not divisible by pow(2, {grid_levels=})'
        assert np.all(grid_levels[:-1] <= grid_levels[1:])
        self._grid_size = grid_size
        self._grid_levels = grid_levels
        self._grid_dims = len(grid_levels)

        # axis sizes
        self._axis_divisions = 2 ** self._grid_levels
        assert len(self._axis_divisions) == self._grid_dims and np.all(grid_size % self._axis_divisions) == 0, 'This should never happen'
        self._axis_division_sizes = grid_size // self._axis_divisions
        
        # info
        self._factor_names = tuple([f'{prefix}-{d}' for prefix in ['color', 'x', 'y'] for d in self._axis_divisions])
        self._factor_sizes = tuple([len(self._colors)] * self._grid_dims + list(self._axis_divisions) * 2)
        print(self._factor_names, self._factor_sizes)
        self._observation_shape = (grid_size, grid_size, 3 if self._rgb else 1)
        
        # initialise
        super().__init__()

    def __getitem__(self, idx):
        positions = self.idx_to_pos(idx)
        cs, xs, ys = positions[:self._grid_dims*1], positions[self._grid_dims*1:self._grid_dims*2], positions[self._grid_dims*2:]
        assert len(xs) == len(ys) == len(cs)
        # GENERATE
        obs = np.full(self.observation_shape, self._bg_color, dtype=np.uint8)
        for i, (x, y, s, c) in enumerate(zip(xs, ys, self._axis_division_sizes, cs)):
            obs[y*s:(y+1)*s, x*s:(x+1)*s, :] = self._colors[c] if np.any(obs[y*s, x*s, :] != self._colors[c]) else self._bg_color
        # RETURN
        return obs


    # @property
    # def factor_sizes(self) -> Tuple[int, ...]:
    #     return self._placements, self._placements, len(self._square_scales), len(self._colors)
    #
    # @property
    # def observation_shape(self) -> Tuple[int, ...]:
    #     return self._width, self._width, (3 if self._rgb else 1)

    # def __init__(self, _grid_size=64, grid_spacing=1, min_square_size=3, max_square_size=9, square_size_spacing=2, rgb=True, palette='colors'):
    #     # generation
    #     self._rgb = rgb
    #     if rgb:
    #         self._colors = np.array(XYData.COLOR_PALETTES_3[palette])
    #     else:
    #         self._colors = np.array(XYData.COLOR_PALETTES_1[palette])
    #     # image sizes
    #     self._width = _grid_size
    #     # square scales
    #     assert min_square_size <= max_square_size
    #     self._max_square_size, self._max_square_size = min_square_size, max_square_size
    #     self._square_scales = np.arange(min_square_size, max_square_size+1, square_size_spacing)
    #     # x, y
    #     self._spacing = grid_spacing
    #     self._placements = (self._width - max_square_size) // grid_spacing + 1
    #     super().__init__()
    
    # def __getitem__(self, idx):
    #     x, y, s, c = self.idx_to_pos(idx)
    #     s = self._square_scales[s]
    #     r = (self._max_square_size - s) // 2
    #     x, y = self._spacing*x + r, self._spacing*y + r
    #     # GENERATE
    #     obs = np.zeros(self.observation_shape, dtype=np.uint8)
    #     obs[y:y+s, x:x+s] = self._colors[c]
    #     return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

# if __name__ == '__main__':
    # data = XYSquaresData(64, [1, 2, 3], rgb=True, palette='rgb', invert_bg=False)        # 110592 // 256 = 432
    # print(len(data))
    # for obs in tqdm(data):
    #     pass
    #     # print(obs[:, :, 0])
