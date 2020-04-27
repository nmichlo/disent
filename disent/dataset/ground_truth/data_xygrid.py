from typing import Tuple

import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from PIL import Image

from disent.dataset.ground_truth.base import GroundTruthData, GroundTruthDataset, PairedVariationDataset
import numpy as np


# ========================================================================= #
# xy grid dataset                                                           #
# ========================================================================= #


class XYData(GroundTruthData):

    """
    Dataset that generates all possible permutations of a point placed on a square grid.
    """

    factor_names = ('x', 'y')
    used_factors = None

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._width, self._width

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._width, self._width

    def __init__(self, width=8):
        self._width = width
        super().__init__()

    def __getitem__(self, idx):
        # GENERATE
        x = np.zeros(self.observation_shape, dtype=np.uint8)
        x[idx // self._width, idx % self._width] = 255  # y, x
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    dataset = GroundTruthDataset(XYData(width=28), transform=torchvision.transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # check access rate
    for i in tqdm(dataset):
        pass

    paired_dataset = PairedVariationDataset(dataset, k='uniform')
    paired_loader = DataLoader(paired_dataset, batch_size=1, shuffle=False)

    # check access rate
    for i in tqdm(paired_loader):
        pass

    # check random factor of variation
    for A, B in tqdm(DataLoader(paired_dataset, batch_size=16, shuffle=True)):
        for a, b in zip(A, B):
            a_pos = dataset.data.idx_to_pos(a.detach().numpy().reshape(28, 28).argmax())
            b_pos = dataset.data.idx_to_pos(b.detach().numpy().reshape(28, 28).argmax())
            print(a_pos, b_pos)
        break