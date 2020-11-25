from abc import abstractmethod
from typing import Optional


class AugmentableDataset(object):

    @property
    @abstractmethod
    def transform(self) -> Optional[callable]:
        raise NotImplementedError

    @property
    @abstractmethod
    def augment(self) -> Optional[callable]:
        raise NotImplementedError

    def _get_augmentable_observation(self, idx):
        raise NotImplementedError

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
        dat = self._get_augmentable_observation(idx)
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

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Multiple Datapoints                                                   #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def dataset_get_observation(self, *idxs):
        xs, x_targs = zip(*[self.dataset_get(idx, mode='pair') for idx in idxs])
        return {
            'x': tuple(xs),
            'x_targ': tuple(x_targs),
        }


def _batch_to_observation(batch, obs_shape):
    """
    Convert a batch of size 1, to a single observation.
    """
    if batch.shape != obs_shape:
        assert batch.shape == (1, *obs_shape)
        return batch.reshape(obs_shape)
    return batch
