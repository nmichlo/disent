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

import dataclasses
import logging
import os
from abc import ABCMeta
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from disent.data.util.hdf5 import hdf5_resave_file
from disent.data.util.hdf5 import PickleH5pyDataset
from disent.data.util.in_out import basename_from_url
from disent.data.util.in_out import ensure_dir_exists
from disent.data.util.in_out import retrieve_file
from disent.data.util.jobs import CachedJobFile
from disent.data.util.state_space import StateSpace


log = logging.getLogger(__name__)


# ========================================================================= #
# ground truth data                                                         #
# ========================================================================= #


class GroundTruthData(StateSpace):

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
# ========================================================================= #


class DiskGroundTruthData(GroundTruthData, metaclass=ABCMeta):

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False):
        super().__init__()
        # get root data folder
        if data_root is None:
            data_root = os.path.abspath(os.environ.get('DISENT_DATA_ROOT', 'data/dataset'))
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
    def data_dir(self):
        return self._data_dir

    @property
    def data_objects(self) -> Sequence['DataObject']:
        raise NotImplementedError


class Hdf5GroundTruthData(DiskGroundTruthData, metaclass=ABCMeta):

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, in_memory=False):
        super().__init__(data_root=data_root, prepare=prepare)
        # variables
        self._in_memory = in_memory
        # load the h5py dataset
        data = PickleH5pyDataset(
            h5_path=self.data_object.get_file_path(self.data_dir),
            h5_dataset_name=self.data_object.hdf5_dataset_name,
        )
        # handle different memroy modes
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
    def data_objects(self) -> Sequence['DlH5DataObject']:
        return [self.data_object]

    @property
    def data_object(self) -> 'DlH5DataObject':
        raise NotImplementedError


# ========================================================================= #
# data objects                                                              #
# ========================================================================= #


class DataObject(object):

    def __init__(self, file_name: str):
        self.file_name = file_name

    def prepare(self, data_dir: str):
        pass

    def get_file_path(self, data_dir: str, variant: Optional[str] = None):
        suffix = '' if (variant is None) else f'.{variant}'
        return os.path.join(data_dir, self.file_name + suffix)


class DlDataObject(DataObject):

    def __init__(
        self,
        # download file/link
        uri: str,
        uri_hash: Union[str, Dict[str, str]],
        # save path
        file_name: Optional[str] = None,  # automatically obtain file name from url if None
        # hash settings
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        super().__init__(file_name=basename_from_url(uri) if (file_name is None) else file_name)
        self.uri = uri
        self.uri_hash = uri_hash
        self.hash_mode = hash_mode
        self.hash_type = hash_type

    def _make_dl_job(self, save_path: str):
        return CachedJobFile(
            make_file_fn=lambda path: retrieve_file(
                src_uri=self.uri,
                dst_path=path,
                overwrite_existing=True,
            ),
            path=save_path,
            hash=self.uri_hash,
            hash_type=self.hash_type,
            hash_mode=self.hash_mode,
        )

    def prepare(self, data_dir: str):
        dl_job = self._make_dl_job(save_path=self.get_file_path(data_dir=data_dir))
        dl_job.run()


class DlH5DataObject(DlDataObject):

    def __init__(
        self,
        # download file/link
        uri: str,
        uri_hash: Union[str, Dict[str, str]],
        # save hash
        file_hash: Union[str, Dict[str, str]],
        # h5 re-save settings
        hdf5_dataset_name: str,
        hdf5_chunk_size: Tuple[int, ...],
        hdf5_compression: Optional[str] = 'gzip',
        hdf5_compression_lvl: Optional[int] = 4,
        hdf5_dtype: Optional[Union[np.dtype, str]] = None,
        hdf5_mutator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        # save path
        file_name: Optional[str] = None,  # automatically obtain file name from url if None
        # hash settings
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        super().__init__(file_name=file_name, uri=uri, uri_hash=uri_hash, hash_mode=hash_mode, hash_type=hash_type)
        self.file_hash = file_hash
        self.hdf5_dataset_name = hdf5_dataset_name
        self.hdf5_chunk_size = hdf5_chunk_size
        self.hdf5_compression = hdf5_compression
        self.hdf5_compression_lvl = hdf5_compression_lvl
        self.hdf5_dtype = hdf5_dtype
        self.hdf5_mutator = hdf5_mutator

    def _make_h5_job(self, load_path: str, save_path: str):
        return CachedJobFile(
            make_file_fn=lambda path: hdf5_resave_file(
                inp_path=load_path,
                out_path=path,
                dataset_name=self.hdf5_dataset_name,
                chunk_size=self.hdf5_chunk_size,
                compression=self.hdf5_compression,
                compression_lvl=self.hdf5_compression_lvl,
                batch_size=None,
                out_dtype=self.hdf5_dtype,
                out_mutator=self.hdf5_mutator,
            ),
            path=save_path,
            hash=self.file_hash,
            hash_type=self.hash_type,
            hash_mode=self.hash_mode,
        )

    def prepare(self, data_dir: str):
        dl_path = self.get_file_path(data_dir=data_dir, variant='ORIG')
        h5_path = self.get_file_path(data_dir=data_dir)
        dl_job = self._make_dl_job(save_path=dl_path)
        h5_job = self._make_h5_job(load_path=dl_path, save_path=h5_path)
        dl_job.set_child(h5_job).run()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #



