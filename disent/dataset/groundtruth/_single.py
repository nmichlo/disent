import torch
from PIL import Image
from torch.utils.data import Dataset
from disent.data.groundtruth.base import GroundTruthData
from disent.util import LengthIter


# ========================================================================= #
# Convert ground truth data to a dataset                                    #
# ========================================================================= #


class GroundTruthDataset(Dataset, LengthIter):
    """
    Converts ground truth data into a dataset
    """

    def __init__(self, ground_truth_data: GroundTruthData, transform=None):
        self.data = ground_truth_data
        # transform observation
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        should return a single observation if an integer index, or
        an array of observations if indices is an array.
        """
        try:
            idx = int(idx)
        except:
            raise TypeError(f'Indices must be integer-like ({type(idx)}): {idx}')

        # we do not support indexing by lists
        image = self.data[idx]

        if self.transform:
            image = self.transform(image)

        return image

    
# ========================================================================= #
# END                                                                       #
# ========================================================================= #
