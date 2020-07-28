from typing import Tuple
from disent.dataset.ground_truth.base import GroundTruthData
import numpy as np


# ========================================================================= #
# xy grid data                                                           #
# ========================================================================= #


class XYScaleData(GroundTruthData):

    """
    Dataset that generates all possible permutations of a square placed on a square grid,
    with varying scale and colour

    - Does not seem to learn with a VAE when square size is equal to 1
      (This property may be explained in the paper "Understanding disentanglement in Beta-VAEs")

    TODO: increase square size
    """

    factor_names = ('x', 'y', 'scale')

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._placements, self._placements, len(self._square_scales)

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._width, self._width

    def __init__(self, grid_size=64, grid_spacing=1, min_square_size=3, max_square_size=9, square_size_spacing=2):
        # image sizes
        self._width = grid_size
        # square scales
        assert min_square_size <= max_square_size
        self._max_square_size, self._max_square_size = min_square_size, max_square_size
        self._square_scales = np.arange(min_square_size, max_square_size+1, square_size_spacing)
        # x, y
        self._spacing = grid_spacing
        self._placements = (self._width - max_square_size) // grid_spacing + 1
        super().__init__()

    def __getitem__(self, idx):
        x, y, s = self.idx_to_pos(idx)
        s = self._square_scales[s]
        r = (self._max_square_size - s) // 2
        x, y = self._spacing*x + r, self._spacing*y + r
        # GENERATE
        obs = np.zeros(self.observation_shape, dtype=np.uint8)
        obs[y:y+s, x:x+s] = 255
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

# if __name__ == '__main__':
#     print(len(XYScaleData()))
#     for i in XYScaleData(6, 2, 2, 4, 2):
#         print(i)
#         print()