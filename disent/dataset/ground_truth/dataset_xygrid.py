from typing import Tuple

import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm

from disent.dataset.ground_truth.base import GroundTruthDataset, PairedVariationDataset
import numpy as np


# ========================================================================= #
# xy grid dataset                                                           #
# ========================================================================= #


class XYDataset(GroundTruthDataset):

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

    def __init__(self, width=8, transform=None):
        self._width = width
        self.transform = transform
        super().__init__()

    def __getitem__(self, idx):
        # GENERATE
        x = np.zeros(self.observation_shape, dtype=np.uint8)
        x[idx % self._width, idx // self._width] = 255  # x, y
        # TRANSFORM
        if self.transform:
            x = self.transform(x)
        return x


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    # dataset = XYDataset(8)
    # print(dataset[dataset.pos_to_idx([2, 1])])

    dataset = XYDataset(width=28, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for i in tqdm(dataset):
        pass

    paired_dataset = PairedVariationDataset(dataset, k='uniform')
    paired_loader = DataLoader(paired_dataset, batch_size=1, shuffle=False)

    for i in tqdm(paired_loader):
        pass