import logging
from typing import Tuple
from disent.dataset.ground_truth_data.base_data import GroundTruthData
import numpy as np


log = logging.getLogger(__name__)


# ========================================================================= #
# xy multi grid data                                                        #
# ========================================================================= #


class XYMultiGridData(GroundTruthData):

    """
    Dataset that generates all possible permutations of 3 (R, G, B) coloured
    squares placed on a square grid.
    
    This dataset is designed to not overlap in the reconstruction loss space.
    (if the spacing is set correctly.)
    """

    factor_names = ('x_R', 'y_R', 'x_G', 'y_G', 'x_B', 'y_B')

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return (self._placements, self._placements) * 3  # R, G, B squares

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, 3

    def __init__(self, square_size=8, grid_size=64, grid_spacing=None):
        if grid_spacing is None:
            grid_spacing = square_size
        if grid_spacing < square_size:
            log.warning(f'overlap between squares for reconstruction loss, {grid_spacing} < {square_size}')
        # image sizes
        self._width = grid_size
        # square scales
        self._square_size = square_size
        # x, y
        self._spacing = grid_spacing
        self._placements = (self._width - self._square_size) // grid_spacing + 1
        self._offset = (self._width - (self._placements * self._spacing)) // 2
        super().__init__()
    
    def __getitem__(self, idx):
        # get factors
        fx0, fy0, fx1, fy1, fx2, fy2 = self.idx_to_pos(idx)
        offset, space, size = self._offset, self._spacing, self._square_size
        # GENERATE
        obs = np.zeros(self.observation_shape, dtype=np.uint8)
        x0, y0 = offset + space * fx0, offset + space * fy0
        x1, y1 = offset + space * fx1, offset + space * fy1
        x2, y2 = offset + space * fx2, offset + space * fy2
        obs[y0:y0+size, x0:x0+size, 0] = 255
        obs[y1:y1+size, x1:x1+size, 1] = 255
        obs[y2:y2+size, x2:x2+size, 2] = 255
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

# if __name__ == '__main__':
#     data = XYMultiGridData(8, 64)
#     print(len(data))  # 262144
#     for i in tqdm(data):
#         pass
#         # print(i[:, :, 0])
#         # print(i[:, :, 1])
#         # print(i[:, :, 2])
#         # print()
