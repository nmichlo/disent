from typing import Tuple
from disent.dataset.util import GroundTruthDataset
from torch.utils.data import Dataset
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
        return self.width, self.width

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self.width, self.width

    def __init__(self, width=8, transform=None, target_transform=None):
        super().__init__()
        self.width = width
        self.data = np.array([self.generate_item(self.width, i)[0] for i in range(self.width**2)])
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    def normalise_pos(self, size, pos):
        return np.array(pos, dtype=np.float32) * (2 / (size - 1)) - 1

    def reverse_pos(self, size, encoding):
        return (np.array(encoding, dtype=np.float32) + 1) / (2 / (size - 1))

    def gen_pair(self, size, idx):
        assert 0 <= idx < (size * size), 'Index out of bounds'
        pos = self.idx_to_pos(size, idx)  # y, x = pos
        encoded = self.normalise_pos(size, pos)  # encoding | range [-1, 1]
        decoded = np.zeros([size, size], dtype=np.float32)  # decoding | range [0, 1]
        decoded[pos[0], pos[1]] = 1
        return encoded, decoded

    def generate_item(self, size, idx, arch='full'):
        encoding, decoding = self.gen_pair(size, idx)
        if arch == 'encoder':
            x, y = decoding, encoding
        elif arch == 'decoder':
            x, y = encoding, idx
        elif arch == 'full':
            x, y = decoding, idx
        else:
            raise KeyError('Invalid arch')
        return x, y


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
