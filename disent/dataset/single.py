from typing import Tuple
import torch
from PIL import Image
from torch.utils.data import Dataset
from disent.dataset.ground_truth_data.base_data import GroundTruthData


# ========================================================================= #
# Convert ground truth data to a dataset                                    #
# ========================================================================= #


class GroundTruthDataset(Dataset, GroundTruthData):
    """
    Converts ground truth data into a dataset
    """

    def __init__(self, ground_truth_data, transform=None):
        self.data = ground_truth_data
        # transform observation
        self.transform = transform
        # initialise GroundTruthData
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        # this takes priority over __getitem__, otherwise __getitem__ would need to
        # raise an IndexError if out of bounds to signal the end of iteration
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, idx):
        """
        should return a single observation if an integer index, or
        an array of observations if indices is an array.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        try:
            idx = int(idx)
        except:
            raise TypeError(f'Indices must be integer-like ({type(idx)}): {idx}')

        # we do not support indexing by lists
        image = self.data[idx]

        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        # PIL Image so that this is consistent with other datasets
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self.data.factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self.data.factor_sizes

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self.data.observation_shape
    
# ========================================================================= #
# END                                                                       #
# ========================================================================= #
