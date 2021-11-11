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
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from torch.utils.data import Dataset

from disent.dataset.util.datafile import DataFile
from disent.dataset.util.datafile import DataFileHashedDlH5
from disent.dataset.data._raw import Hdf5Dataset
from disent.dataset.util.state_space import StateSpace
from disent.util.inout.paths import ensure_dir_exists


log = logging.getLogger(__name__)


# ========================================================================= #
# ground truth data                                                         #
# ========================================================================= #


class GroundTruthData(Dataset, StateSpace):
    """
    Dataset that corresponds to some state space or ground truth factors
    """

    def __init__(self, transform=None):
        self._transform = transform
        super().__init__(
            factor_sizes=self.factor_sizes,
            factor_names=self.factor_names,
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Overridable Defaults                                                  #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def name(self):
        name = self.__class__.__name__
        if name.endswith('Data'):
            name = name[:-len('Data')]
        return name.lower()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # State Space                                                           #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def factor_names(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Properties                                                            #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def x_shape(self) -> Tuple[int, ...]:
        # shape as would be for a single observation in a torch batch
        # eg. C x H x W
        H, W, C = self.img_shape
        return (C, H, W)

    @property
    def img_shape(self) -> Tuple[int, ...]:
        # shape as would be for an original image
        # eg. H x W x C
        raise NotImplementedError()

    @property
    def img_channels(self) -> int:
        channels = self.img_shape[-1]
        assert channels in (1, 3), f'invalid number of channels for dataset: {self.__class__.__name__}, got: {repr(channels)}, required: 1 or 3'
        return channels

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Overrides                                                             #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def __getitem__(self, idx):
        obs = self._get_observation(idx)
        if self._transform is not None:
            obs = self._transform(obs)
        return obs

    def _get_observation(self, idx):
        raise NotImplementedError

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # EXTRAS                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def sample_random_obs_traversal(self, f_idx: int = None, base_factors=None, num: int = None, mode='interval', obs_collect_fn=None) -> Tuple[np.ndarray, np.ndarray, Union[List[Any], Any]]:
        """
        Same API as sample_random_factor_traversal, but also
        returns the corresponding indices and uncollated list of observations
        """
        factors = self.sample_random_factor_traversal(f_idx=f_idx, base_factors=base_factors, num=num, mode=mode)
        indices = self.pos_to_idx(factors)
        obs = [self[i] for i in indices]
        if obs_collect_fn is not None:
            obs = obs_collect_fn(obs)
        return factors, indices, obs


# ========================================================================= #
# Basic Array Ground Truth Dataset                                          #
# ========================================================================= #


class ArrayGroundTruthData(GroundTruthData):

    def __init__(self, array, factor_names: Tuple[str, ...], factor_sizes: Tuple[int, ...], array_chn_is_last: bool = True, x_shape: Optional[Tuple[int, ...]] = None, transform=None):
        self.__factor_names = tuple(factor_names)
        self.__factor_sizes = tuple(factor_sizes)
        self._array = array
        # get shape
        if x_shape is not None:
            C, H, W = x_shape
        elif array_chn_is_last:
            H, W, C = array.shape[1:]
        else:
            C, H, W = array.shape[1:]
        # set observation shape
        self.__img_shape = (H, W, C)
        # initialize
        super().__init__(transform=transform)
        # check shapes -- it is up to the user to handle which method they choose
        assert (array.shape[1:] == self.img_shape) or (array.shape[1:] == self.x_shape)

    @property
    def array(self):
        return self._array

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self.__factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self.__factor_sizes

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self.__img_shape

    def _get_observation(self, idx):
        # TODO: INVESTIGATE! I think this implements a lock,
        #       hindering multi-threaded environments?
        return self._array[idx]

    @classmethod
    def new_like(cls, array, gt_data: GroundTruthData, array_chn_is_last: bool = True):
        # TODO: should this not copy the x_shape and transform?
        return cls(
            array=array,
            factor_names=gt_data.factor_names,
            factor_sizes=gt_data.factor_sizes,
            array_chn_is_last=array_chn_is_last,
            x_shape=None,  # infer from array
            transform=None,
        )


# ========================================================================= #
# disk ground truth data                                                    #
# TODO: data & datafile preparation should be split out from                #
#       GroundTruthData, instead GroundTruthData should be a wrapper        #
# ========================================================================= #


class _DiskDataMixin(object):

    # attr this class defines in _mixin_disk_init
    _data_dir: str

    def _mixin_disk_init(self, data_root: Optional[str] = None, prepare: bool = False):
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
                log.debug(f'[preparing]: {datafile} into data dir: {self._data_dir}')
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

    @property
    def name(self) -> str:
        raise NotImplementedError


class DiskGroundTruthData(_DiskDataMixin, GroundTruthData, metaclass=ABCMeta):

    """
    Dataset that prepares a list DataObjects into some local directory.
    - This directory can be
    """

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, transform=None):
        super().__init__(transform=transform)
        # get root data folder
        self._mixin_disk_init(data_root=data_root, prepare=prepare)


class NumpyFileGroundTruthData(DiskGroundTruthData, metaclass=ABCMeta):
    """
    Dataset that loads a numpy file from a DataObject
    - if the dataset is contained in a key, set the `data_key` property
    """

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, transform=None):
        super().__init__(data_root=data_root, prepare=prepare, transform=transform)
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

    def _get_observation(self, idx):
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


class _Hdf5DataMixin(object):

    # attrs this class defines in _mixin_hdf5_init
    _in_memory: bool
    _attrs: dict
    _data: Union[Hdf5Dataset, np.ndarray]

    def _mixin_hdf5_init(self, h5_path: str, h5_dataset_name: str = 'data', in_memory: bool = False):
        # variables
        self._in_memory = in_memory
        # load the h5py dataset
        data = Hdf5Dataset(
            h5_path=h5_path,
            h5_dataset_name=h5_dataset_name,
        )
        # load attributes
        self._attrs = data.get_attrs()
        # handle different memory modes
        if self._in_memory:
            # Load the entire dataset into memory if required
            # indexing dataset objects returns numpy array
            # instantiating np.array from the dataset requires double memory.
            self._data = data[:]
            self._data.flags.writeable = False
            data.close()
        else:
            # Load the dataset from the disk
            self._data = data

    def __len__(self):
        return len(self._data)

    @property
    def img_shape(self):
        shape = self._data.shape[1:]
        assert len(shape) == 3
        return shape

    # override from GroundTruthData
    def _get_observation(self, idx):
        return self._data[idx]


class Hdf5GroundTruthData(_Hdf5DataMixin, DiskGroundTruthData, metaclass=ABCMeta):
    """
    Dataset that loads an Hdf5 file from a DataObject
    - requires that the data object has the `out_dataset_name` attribute
      that points to the hdf5 dataset in the file to load.
    """

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, in_memory=False, transform=None):
        super().__init__(data_root=data_root, prepare=prepare, transform=transform)
        # initialize mixin
        self._mixin_hdf5_init(
            h5_path=os.path.join(self.data_dir, self.datafile.out_name),
            h5_dataset_name=self.datafile.dataset_name,
            in_memory=in_memory,
        )

    @property
    def datafiles(self) -> Sequence[DataFileHashedDlH5]:
        return [self.datafile]

    @property
    def datafile(self) -> DataFileHashedDlH5:
        raise NotImplementedError


class SelfContainedHdf5GroundTruthData(_Hdf5DataMixin, GroundTruthData):

    def __init__(self, h5_path: str, in_memory=False, transform=None):
        # initialize mixin
        self._mixin_hdf5_init(
            h5_path=h5_path,
            h5_dataset_name='data',
            in_memory=in_memory,
        )
        # load attrs
        self._attr_name = self._attrs['dataset_name'].decode("utf-8")
        self._attr_factor_names = tuple(name.decode("utf-8") for name in self._attrs['factor_names'])
        self._attr_factor_sizes = tuple(int(size) for size in self._attrs['factor_sizes'])
        # set size
        (B, H, W, C) = self._data.shape
        self._img_shape = (H, W, C)
        # initialize!
        super().__init__(transform=transform)

    @property
    def name(self) -> str:
        return self._attr_name

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self._attr_factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self._attr_factor_sizes

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self._img_shape


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
