from typing import Tuple
from torch.utils.data import Dataset
import numpy as np
from disent.dataset.ground_truth.ground_truth import GroundTruthData


# ========================================================================= #
# xy grid dataset                                                           #
# ========================================================================= #


class XYDataset(Dataset, GroundTruthData):

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
        self.data = np.array([XYDataset.generate_item(self.width, i)[0] for i in range(self.width**2)])
        self.transform = transform
        self.target_transform = target_transform

    def get_observations_from_indices(self, indices):
        return self.data[indices]

    def __len__(self):
        return self.size*self.size

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y

    @staticmethod
    def normalise_pos(size, pos):
        return np.array(pos, dtype=np.float32) * (2 / (size - 1)) - 1

    @staticmethod
    def reverse_pos(size, encoding):
        return (np.array(encoding, dtype=np.float32) + 1) / (2 / (size - 1))

    @staticmethod
    def pos2idx(size, pos):
        return int(pos[0] * size + pos[1])

    @staticmethod
    def idx2pos(size, idx):
        return [idx // size, idx % size]

    @staticmethod
    def gen_pair(size, idx):
        assert 0 <= idx < (size * size), 'Index out of bounds'
        pos = XYDataset.idx2pos(size, idx)  # y, x = pos
        encoded = XYDataset.normalise_pos(size, pos)  # encoding | range [-1, 1]
        decoded = np.zeros([size, size], dtype=np.float32)  # decoding | range [0, 1]
        decoded[pos[0], pos[1]] = 1
        return encoded, decoded

    @staticmethod
    def generate_item(size, idx, arch='full'):
        encoding, decoding = XYDataset.gen_pair(size, idx)
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
