import logging
from typing import Tuple

from torch.utils.data import Dataset
from disent.data.groundtruth.base import GroundTruthData


log = logging.getLogger(__name__)


# ========================================================================= #
# Convert ground truth data to a dataset                                    #
# ========================================================================= #


class GroundTruthDataset(Dataset, GroundTruthData):

    def __init__(self, ground_truth_data: GroundTruthData, transform=None, augment=None):
        assert isinstance(ground_truth_data, GroundTruthData), f'{ground_truth_data=} must be an instance of GroundTruthData!'
        self.data = ground_truth_data
        super().__init__()
        self.transform = transform
        self.augment = augment

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # State Space Overrides                                                 #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self.data.factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self.data.factor_sizes

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self.data.observation_shape

    def __getitem__(self, idx):
        x0, x0_targ = self.datapoint_get_input_target_pair(idx)
        return {
            'x': (x0,),            # wrapped in tuple to match pair and triplet
            'x_targ': (x0_targ,),  # wrapped in tuple to match pair and triplet
        }

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Single Datapoints                                                     #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _datapoint_raw_to_target(self, dat):
        x_targ = dat
        if self.transform:
            x_targ = self.transform(x_targ)
        return x_targ

    def _datapoint_target_to_input(self, x_targ):
        x = x_targ
        if self.augment:
            x = self.augment(x)
            x = _batch_to_observation(batch=x, obs_shape=x_targ.shape)
        return x

    def datapoint_get_raw(self, idx):
        try:
            idx = int(idx)
        except:
            raise TypeError(f'Indices must be integer-like ({type(idx)}): {idx}')
        # we do not support indexing by lists
        return self.data[idx]

    def datapoint_get_target(self, idx):
        dat = self.datapoint_get_raw(idx)
        x_targ = self._datapoint_raw_to_target(dat)
        return x_targ

    def datapoint_get_input(self, idx):
        x, x_targ = self.datapoint_get_input_target_pair(idx)
        return x

    def datapoint_get_input_target_pair(self, idx):
        dat = self.datapoint_get_raw(idx)
        x_targ = self._datapoint_raw_to_target(dat)
        x = self._datapoint_target_to_input(x_targ)
        return x, x_targ

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # End Class                                                             #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


def _batch_to_observation(batch, obs_shape):
    if batch.shape != obs_shape:
        assert batch.shape == (1, *obs_shape)
        return batch.reshape(obs_shape)
    return batch


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
