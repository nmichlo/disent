#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import logging
import os
from abc import ABCMeta
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from disent.data.dataobj import DataObject
from disent.data.dataobj import DlH5DataObject
from disent.data.groundtruth.states import StateSpace
from disent.data.hdf5 import PickleH5pyData
from disent.util.paths import ensure_dir_exists


log = logging.getLogger(__name__)


# ========================================================================= #
# ground truth data                                                         #
# ========================================================================= #


class GroundTruthData(StateSpace):
    """
    Dataset that corresponds to some state space or ground truth factors
    """

    def __init__(self):
        assert len(self.factor_names) == len(self.factor_sizes), 'Dimensionality mismatch of FACTOR_NAMES and FACTOR_DIMS'
        super().__init__(factor_sizes=self.factor_sizes)

    @property
    def name(self):
        name = self.__class__.__name__
        if name.endswith('Data'):
            name = name[:-len('Data')]
        return name.lower()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Overrides                                                             #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def factor_names(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        # shape as would be for a non-batched observation
        # eg. H x W x C
        raise NotImplementedError()

    @property
    def x_shape(self) -> Tuple[int, ...]:
        # shape as would be for a single observation in a torch batch
        # eg. C x H x W
        shape = self.observation_shape
        return shape[-1], *shape[:-1]

    def __getitem__(self, idx):
        raise NotImplementedError


# ========================================================================= #
# disk ground truth data                                                    #
# TODO: data & data_object preparation should be split out from             #
#       GroundTruthData, instead GroundTruthData should be a wrapper        #
# ========================================================================= #


class DiskGroundTruthData(GroundTruthData, metaclass=ABCMeta):

    """
    Dataset that prepares a list DataObjects into some local directory.
    - This directory can be
    """

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False):
        super().__init__()
        # get root data folder
        if data_root is None:
            data_root = self.default_data_root
        else:
            data_root = os.path.abspath(data_root)
        # get class data folder
        self._data_dir = ensure_dir_exists(os.path.join(data_root, self.name))
        log.info(f'{self.name}: data_dir_share={repr(self._data_dir)}')
        # prepare everything
        if prepare:
            for data_object in self.data_objects:
                data_object.prepare(self.data_dir)

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def default_data_root(self):
        return os.path.abspath(os.environ.get('DISENT_DATA_ROOT', 'data/dataset'))

    @property
    def data_objects(self) -> Sequence[DataObject]:
        raise NotImplementedError


class NumpyGroundTruthData(DiskGroundTruthData, metaclass=ABCMeta):
    """
    Dataset that loads a numpy file from a DataObject
    - if the dataset is contained in a key, set the `data_key` property
    """

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False):
        super().__init__(data_root=data_root, prepare=prepare)
        # load dataset
        self._data = np.load(os.path.join(self.data_dir, self.data_object.out_name))
        if self.data_key is not None:
            self._data = self._data[self.data_key]

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def data_objects(self) -> Sequence[DataObject]:
        return [self.data_object]

    @property
    def data_object(self) -> DataObject:
        raise NotImplementedError

    @property
    def data_key(self) -> Optional[str]:
        # can override this!
        return None


class Hdf5GroundTruthData(DiskGroundTruthData, metaclass=ABCMeta):
    """
    Dataset that loads an Hdf5 file from a DataObject
    - requires that the data object has the `out_dataset_name` attribute
      that points to the hdf5 dataset in the file to load.
    """

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, in_memory=False):
        super().__init__(data_root=data_root, prepare=prepare)
        # variables
        self._in_memory = in_memory
        # load the h5py dataset
        data = PickleH5pyData(
            h5_path=os.path.join(self.data_dir, self.data_object.out_name),
            h5_dataset_name=self.data_object.out_dataset_name,
        )
        # handle different memory modes
        if self._in_memory:
            # Load the entire dataset into memory if required
            # indexing dataset objects returns numpy array
            # instantiating np.array from the dataset requires double memory.
            self._data = data[:]
            data.close()
        else:
            # Load the dataset from the disk
            self._data = data

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def data_objects(self) -> Sequence[DlH5DataObject]:
        return [self.data_object]

    @property
    def data_object(self) -> DlH5DataObject:
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

