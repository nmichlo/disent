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
        assert isinstance(ground_truth_data, GroundTruthData), f'{ground_truth_data=} must be an instance of GroundTruthData!'
        self.data = ground_truth_data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self._getitem_transformed(idx)

    def _getitem_transformed(self, idx):
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
