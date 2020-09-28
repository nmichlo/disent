import kornia
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

    def __init__(self, ground_truth_data: GroundTruthData, transform=None, augment=None):
        assert isinstance(ground_truth_data, GroundTruthData), f'{ground_truth_data=} must be an instance of GroundTruthData!'
        self.data = ground_truth_data
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x0, x0_targ = self._getitem_transformed(idx)
        return {
            'x': x0,
            'x_targ': x0_targ,
        }

    def _getitem_transformed(self, idx):
        try:
            idx = int(idx)
        except:
            raise TypeError(f'Indices must be integer-like ({type(idx)}): {idx}')

        # we do not support indexing by lists
        obs = self.data[idx]

        x_targ = obs
        if self.transform:
            x_targ = self.transform(x_targ)

        # get target data
        x = x_targ
        if self.augment:
            x = self.augment(x_targ)

            # TODO: temp! this should not be here...
            #       kornia augmentations are meant to be applied after being converted to a batch
            #       so we need to remove the extra dimension
            #       Move augmentations elsewhere...
            if x_targ.shape != x.shape:
                assert len(x_targ.shape) < len(x.shape)
                assert x.shape[0] == 1
                x = x.reshape(x.shape[1:])

        return x, x_targ


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
