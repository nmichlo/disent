from torch.utils.data import Dataset
import numpy as np


# ========================================================================= #
# xy grid dataset                                                           #
# ========================================================================= #


class XYDataset(Dataset):

    def __init__(self, size=8, arch='full', transform=None, target_transform=None):
        self.size = size
        self.arch = arch

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.size*self.size

    def __getitem__(self, idx):
        encoding, decoding = XYDataset.gen_pair(self.size, idx)
        if self.arch == 'encoder':
            x, y = decoding, encoding
        elif self.arch == 'decoder':
            x, y = encoding, idx
        elif self.arch == 'full':
            x, y = decoding, idx
        else:
            raise KeyError('Invalid arch')

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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
