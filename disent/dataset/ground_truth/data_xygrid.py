from typing import Tuple
from disent.dataset.ground_truth.base import GroundTruthData
import numpy as np


# ========================================================================= #
# xy grid data                                                           #
# ========================================================================= #


class XYData(GroundTruthData):

    """
    Dataset that generates all possible permutations of a point placed on a square grid.

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

    def __init__(self, grid_size=9, square_size=2):
        self._width = grid_size  # image size
        self._square_width = square_size  # square size
        self._placements = self._width - (self._square_width - 1)
        super().__init__()

    def __getitem__(self, idx):
        # GENERATE
        obs = np.zeros(self.observation_shape, dtype=np.uint8)
        y, x = idx // self._placements, idx % self._placements
        obs[y:y+self._square_width, x:x+self._square_width] = 255  # y, x
        return obs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

# if __name__ == '__main__':
#     dataset = GroundTruthDataset(XYData(width=28), transform=torchvision.transforms.ToTensor())
#     loader = DataLoader(dataset, batch_size=1, shuffle=False)
#
#     # check access rate
#     for i in tqdm(dataset):
#         pass
#
#     paired_dataset = PairedVariationDataset(dataset, k='uniform')
#     paired_loader = DataLoader(paired_dataset, batch_size=1, shuffle=False)
#
#     # check access rate
#     for i in tqdm(paired_loader):
#         pass
#
#     # check random factor of variation
#     for A, B in tqdm(DataLoader(paired_dataset, batch_size=16, shuffle=True)):
#         for a, b in zip(A, B):
#             a_pos = dataset.data.idx_to_pos(a.detach().numpy().reshape(28, 28).argmax())
#             b_pos = dataset.data.idx_to_pos(b.detach().numpy().reshape(28, 28).argmax())
#             print(a_pos, b_pos)
#         break