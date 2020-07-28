from typing import Tuple
from disent.dataset.ground_truth.base import GroundTruthData
import numpy as np


# ========================================================================= #
# xy grid data                                                           #
# ========================================================================= #


class XYData(GroundTruthData):

    """
    Dataset that generates all possible permutations of a square placed on a square grid.

    - Does not seem to learn with a VAE when square size is equal to 1
      (This property may be explained in the paper "Understanding disentanglement in Beta-VAEs")

    TODO: increase square size
    """

    factor_names = ('x', 'y')

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._placements, self._placements

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._width, self._width

    def __init__(self, grid_size=64, grid_spacing=1, square_size=5):
        self._width = grid_size  # image size
        self._square_width = square_size  # square size
        # x, y
        self._spacing = grid_spacing
        self._placements = (self._width - square_size) // grid_spacing + 1
        super().__init__()

    def __getitem__(self, idx):
        x, y = self.idx_to_pos(idx)
        x, y = self._spacing*x, self._spacing*y
        # GENERATE
        obs = np.zeros(self.observation_shape, dtype=np.uint8)
        obs[y:y+self._square_width, x:x+self._square_width] = 255
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

# if __name__ == '__main__':
#     print(len(XYData()))
#     for i in XYData(6, 2, 3):
#         print(i)
#         print()