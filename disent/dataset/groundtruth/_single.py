import logging
from typing import Tuple
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
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
        x0, x0_targ = self.dataset_get(idx, mode='pair')
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

    def dataset_get(self, idx, mode: str):
        try:
            idx = int(idx)
        except:
            raise TypeError(f'Indices must be integer-like ({type(idx)}): {idx}')
        # we do not support indexing by lists
        dat = self.data[idx]
        # return correct data
        if mode == 'pair':
            x_targ = self._datapoint_raw_to_target(dat)
            x = self._datapoint_target_to_input(x_targ)
            return x, x_targ
        elif mode == 'input':
            x_targ = self._datapoint_raw_to_target(dat)
            return self._datapoint_target_to_input(x_targ)
        elif mode == 'target':
            return self._datapoint_raw_to_target(dat)
        elif mode == 'raw':
            return dat
        else:
            raise KeyError(f'Invalid {mode=}')

    def dataset_batch_from_factors(self, factors: np.ndarray, mode: str):
        """Get a batch of observations X from a batch of factors Y."""
        return default_collate([
            self.dataset_get(idx, mode=mode)
            for idx in self.pos_to_idx(factors)
        ])

    def dataset_sample_batch_with_factors(self, num_samples: int, mode: str):
        """Sample a batch of observations X and factors Y."""
        factors = self.sample_factors(num_samples)
        batch = self.dataset_batch_from_factors(factors, mode=mode)
        return batch, default_collate(factors)

    def dataset_sample_batch(self, num_samples: int, mode: str):
        """Sample a batch of observations X."""
        factors = self.sample_factors(num_samples)
        batch = self.dataset_batch_from_factors(factors, mode=mode)
        return batch

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
