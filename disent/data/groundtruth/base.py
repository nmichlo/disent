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

from disent.data.datafile import DataFile
from disent.data.datafile import DataFileHashedDlH5
from disent.data.groundtruth.states import StateSpace
from disent.data.hdf5 import PickleH5pyFile
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
        super().__init__(
            factor_sizes=self.factor_sizes,
            factor_names=self.factor_names,
        )

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
# TODO: data & datafile preparation should be split out from             #
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
            for datafile in self.datafiles:
                datafile.prepare(self.data_dir)

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @property
    def default_data_root(self):
        return os.path.abspath(os.environ.get('DISENT_DATA_ROOT', 'data/dataset'))

    @property
    def datafiles(self) -> Sequence[DataFile]:
        raise NotImplementedError


class NumpyGroundTruthData(DiskGroundTruthData, metaclass=ABCMeta):
    """
    Dataset that loads a numpy file from a DataObject
    - if the dataset is contained in a key, set the `data_key` property
    """

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False):
        super().__init__(data_root=data_root, prepare=prepare)
        # load dataset
        load_path = os.path.join(self.data_dir, self.datafile.out_name)
        if load_path.endswith('.gz'):
            import gzip
            with gzip.GzipFile(load_path, 'r') as load_file:
                self._data = np.load(load_file)
        else:
            self._data = np.load(load_path)
        # load from the key if specified
        if self.data_key is not None:
            self._data = self._data[self.data_key]

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def datafiles(self) -> Sequence[DataFile]:
        return [self.datafile]

    @property
    def datafile(self) -> DataFile:
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
        data = PickleH5pyFile(
            h5_path=os.path.join(self.data_dir, self.datafile.out_name),
            h5_dataset_name=self.datafile.dataset_name,
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
    def datafiles(self) -> Sequence[DataFileHashedDlH5]:
        return [self.datafile]

    @property
    def datafile(self) -> DataFileHashedDlH5:
        raise NotImplementedError


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

