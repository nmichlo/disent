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
from typing import Callable
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from disent.data.util.hdf5 import hdf5_resave_file
from disent.data.util.hdf5 import PickleH5pyDataset
from disent.data.util.in_out import basename_from_url
from disent.data.util.in_out import modify_file_name
from disent.data.util.in_out import stalefile
from disent.data.util.in_out import download_file
from disent.data.util.in_out import ensure_dir_exists
from disent.data.util.in_out import retrieve_file
from disent.data.util.state_space import StateSpace
from disent.util import wrapped_partial


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
    def data_objects(self) -> Sequence['DataObject']:
        raise NotImplementedError


class NumpyGroundTruthData(DiskGroundTruthData, metaclass=ABCMeta):

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False):
        super().__init__(data_root=data_root, prepare=prepare)
        # load dataset
        self._data = np.load(os.path.join(self.data_dir, self.data_object.out_name))
        if self.data_key is not None:
            self._data = self._data[self.data_key]

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def data_objects(self) -> Sequence['DataObject']:
        return [self.data_object]

    @property
    def data_object(self) -> 'DataObject':
        raise NotImplementedError

    @property
    def data_key(self) -> Optional[str]:
        # can override this!
        return None


class Hdf5GroundTruthData(DiskGroundTruthData, metaclass=ABCMeta):

    def __init__(self, data_root: Optional[str] = None, prepare: bool = False, in_memory=False):
        super().__init__(data_root=data_root, prepare=prepare)
        # variables
        self._in_memory = in_memory
        # load the h5py dataset
        data = PickleH5pyDataset(
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
    def data_objects(self) -> Sequence['DlH5DataObject']:
        return [self.data_object]

    @property
    def data_object(self) -> 'DlH5DataObject':
        raise NotImplementedError


# ========================================================================= #
# data objects                                                              #
# TODO: clean this up, this could be simplified!                            #
# ========================================================================= #


class DataObject(object):

    def __init__(
        self,
        file_name: str,
        file_hash: Optional[Union[str, Dict[str, str]]],
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        self._file_name = file_name
        self._file_hash = file_hash
        self._hash_type = hash_type
        self._hash_mode = hash_mode

    @property
    def out_name(self) -> str:
        return self._file_name

    def prepare(self, out_dir: str) -> str:
        @stalefile(file=os.path.join(out_dir, self._file_name), hash=self._file_hash, hash_type=self._hash_type, hash_mode=self._hash_mode)
        def wrapped(out_file):
            self._prepare(out_dir=out_dir, out_file=out_file)
        return wrapped()

    def _prepare(self, out_dir: str, out_file: str) -> str:
        raise NotImplementedError


class DlDataObject(DataObject):

    def __init__(
        self,
        uri: str,
        uri_hash: Optional[Union[str, Dict[str, str]]],
        uri_name: Optional[str] = None,
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        super().__init__(
            file_name=basename_from_url(uri) if (uri_name is None) else uri_name,
            file_hash=uri_hash,
            hash_type=hash_type,
            hash_mode=hash_mode
        )
        self._uri = uri

    def _prepare(self, out_dir: str, out_file: str):
        retrieve_file(src_uri=self._uri, dst_path=out_file, overwrite_existing=True)


class DlGenDataObject(DataObject):

    def __init__(
        self,
        # download & save files
        uri: str,
        uri_hash: Optional[Union[str, Dict[str, str]]],
        file_hash: Optional[Union[str, Dict[str, str]]],
        # save paths
        uri_name: Optional[str] = None,
        file_name: Optional[str] = None,
        # hash settings
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        self._dl_obj = DlDataObject(
            uri=uri,
            uri_hash=uri_hash,
            uri_name=uri_name,
            hash_type=hash_type,
            hash_mode=hash_mode,
        )
        super().__init__(
            file_name=modify_file_name(self._dl_obj.out_name, prefix='gen') if (file_name is None) else file_name,
            file_hash=file_hash,
            hash_type=hash_type,
            hash_mode=hash_mode,
        )

    def _prepare(self, out_dir: str, out_file: str):
        inp_file = self._dl_obj.prepare(out_dir=out_dir)
        self._generate(inp_file=inp_file, out_file=out_file)

    def _generate(self, inp_file: str, out_file: str):
        raise NotImplementedError


class DlH5DataObject(DlGenDataObject):

    def __init__(
        self,
        # download & save files
        uri: str,
        uri_hash: Optional[Union[str, Dict[str, str]]],
        file_hash: Optional[Union[str, Dict[str, str]]],
        # h5 re-save settings
        hdf5_dataset_name: str,
        hdf5_chunk_size: Tuple[int, ...],
        hdf5_compression: Optional[str] = 'gzip',
        hdf5_compression_lvl: Optional[int] = 4,
        hdf5_dtype: Optional[Union[np.dtype, str]] = None,
        hdf5_mutator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        # save paths
        uri_name: Optional[str] = None,
        file_name: Optional[str] = None,
        # hash settings
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        super().__init__(
            file_name=file_name,
            file_hash=file_hash,
            uri=uri,
            uri_hash=uri_hash,
            uri_name=uri_name,
            hash_type=hash_type,
            hash_mode=hash_mode,
        )
        self._hdf5_resave_file = wrapped_partial(
            hdf5_resave_file,
            dataset_name=hdf5_dataset_name,
            chunk_size=hdf5_chunk_size,
            compression=hdf5_compression,
            compression_lvl=hdf5_compression_lvl,
            out_dtype=hdf5_dtype,
            out_mutator=hdf5_mutator,
        )
        # save the dataset name
        self._out_dataset_name = hdf5_dataset_name

    @property
    def out_dataset_name(self) -> str:
        return self._out_dataset_name

    def _generate(self, inp_file: str, out_file: str):
        self._hdf5_resave_file(inp_path=inp_file, out_path=out_file)


# class DataObject(object):
#
#     @property
#     def file_name(self) -> str:
#         raise NotImplementedError
#
#     def prepare(self, data_dir: str):
#         pass
#
#     def get_path(self, data_dir, *attrs):
#         paths = [os.path.join(data_dir, getattr(self, attr)) for attr in attrs]
#         if len(paths) == 1:
#             return paths[0]
#         return paths
#
#
# class DlDataObject(DataObject):
#
#     def __init__(
#         self,
#         # download file/link
#         uri: str,
#         uri_hash: Optional[Union[str, Dict[str, str]]],
#         uri_name: Optional[str] = None,  # automatically obtain uri name from url if None
#         # hash settings
#         hash_type: str = 'md5',
#         hash_mode: str = 'fast',
#     ):
#         def _prepare(data_dir: str):
#             dl_path = os.path.join(data_dir, self._uri_name)
#             # cached download task
#             @stalefile(file=dl_path, hash=uri_hash, hash_type=hash_type, hash_mode=hash_mode)
#             def download():
#                 retrieve_file(src_uri=uri, dst_path=dl_path, overwrite_existing=True)
#             # run task
#             return download()
#
#         # instance variables
#         self._uri_name = basename_from_url(uri) if (uri_name is None) else uri_name
#         self._out_name = self._uri_name
#         self._prepare = _prepare
#
#     @property
#     def file_name(self) -> str:
#         return self._out_name
#
#     def prepare(self, data_dir: str):
#         return self._prepare(data_dir)
#
#
# class DlGenDataObject(DlDataObject):
#
#     def __init__(
#         self,
#         # save path
#         file_name: str,
#         file_hash: Optional[Union[str, Dict[str, str]]],
#         # download file/link
#         uri: str,
#         uri_hash: Optional[Union[str, Dict[str, str]]],
#         uri_name: Optional[str] = None,  # automatically obtain file name from url if None
#         # hash settings
#         hash_type: str = 'md5',
#         hash_mode: str = 'fast',
#     ):
#         super().__init__(uri=uri, uri_hash=uri_hash, uri_name=uri_name, hash_mode=hash_mode, hash_type=hash_type)
#
#         def _prepare(data_dir: str):
#             proc_path = os.path.join(data_dir, file_name)
#             # cached process task
#             @stalefile(file=proc_path, hash=file_hash, hash_type=hash_type, hash_mode=hash_mode)
#             def process():
#                 self._process_file(inp_file=self._prepare_dl(data_dir), out_file=proc_path)
#             # run task
#             return process()
#
#         self._prepare, self._prepare_dl = _prepare, self._prepare
#         self._out_name = file_name
#
#     def _process_file(self, inp_file: str, out_file: str):
#         raise NotImplementedError
#
#
# class DlH5DataObject(DlGenDataObject):
#
#     def __init__(
#         self,
#         # download file/link
#         uri: str,
#         uri_hash: Optional[Union[str, Dict[str, str]]],
#         # save hash
#         file_hash: Optional[Union[str, Dict[str, str]]],
#         # h5 re-save settings
#         hdf5_dataset_name: str,
#         hdf5_chunk_size: Tuple[int, ...],
#         hdf5_compression: Optional[str] = 'gzip',
#         hdf5_compression_lvl: Optional[int] = 4,
#         hdf5_dtype: Optional[Union[np.dtype, str]] = None,
#         hdf5_mutator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
#         # save path
#         file_name: Optional[str] = None,  # automatically obtain file name from url if None
#         # hash settings
#         hash_type: str = 'md5',
#         hash_mode: str = 'fast',
#     ):
#         file_name = basename_from_url(uri) if (file_name is None) else file_name
#         uri_name = f'dl.{file_name}'
#         self.hdf5_dataset_name = hdf5_dataset_name
#
#         super().__init__(
#             file_name=file_name,
#             file_hash=file_hash,
#             uri=uri,
#             uri_hash=uri_hash,
#             uri_name=uri_name,
#             hash_type=hash_type,
#             hash_mode=hash_mode
#         )
#
#         def prepare_file(inp_file: str, out_file: str):
#             hdf5_resave_file(
#                 inp_path=inp_file,
#                 out_path=out_file,
#                 dataset_name=hdf5_dataset_name,
#                 chunk_size=hdf5_chunk_size,
#                 compression=hdf5_compression,
#                 compression_lvl=hdf5_compression_lvl,
#                 batch_size=None,
#                 out_dtype=hdf5_dtype,
#                 out_mutator=hdf5_mutator,
#             )
#
#         self._process_file = prepare_file


DataFilePrepare = Callable[[str, str], NoReturn]
DataPrepare = Callable[[str], str]


def stalefile_prepare(
    out_name,
    out_hash: Union[str, Dict[str, str]],
    hash_type: str = 'md5',
    hash_mode: str = 'fast',
) -> Callable[[DataFilePrepare], DataPrepare]:
    def wrapper(func: DataFilePrepare) -> DataPrepare:
        def prepare(data_dir: str) -> str:
            out_file = os.path.join(data_dir, out_name)
            @stalefile(file=out_file, hash=out_hash, hash_type=hash_type, hash_mode=hash_mode)
            def run():
                func(data_dir, out_file)
            return run()
        return prepare
    return wrapper


# def data_object_downloader(
#     uri: str,
#     uri_hash: Optional[Union[str, Dict[str, str]]],
#     uri_name: Optional[str] = None,
#     hash_type: str = 'md5',
#     hash_mode: str = 'fast',
# ) -> DataObject:
#
#     uri_name = basename_from_url(uri) if (uri_name is None) else uri_name
#
#     @prepare_stalefile(out_name=uri_name, out_hash=uri_hash, hash_type=hash_type, hash_mode=hash_mode)
#     def download(out_dir: str, out_path: str):
#         retrieve_file(src_uri=uri, dst_path=out_path, overwrite_existing=True)
#         return out_path
#
#     return download
#
#
# def data_object_download_h5(
#     # save hash
#     file_hash: Optional[Union[str, Dict[str, str]]],
#     # download file/link
#     uri: str,
#     uri_hash: Optional[Union[str, Dict[str, str]]],
#     # h5 re-save settings
#     hdf5_dataset_name: str,
#     hdf5_chunk_size: Tuple[int, ...],
#     hdf5_compression: Optional[str] = 'gzip',
#     hdf5_compression_lvl: Optional[int] = 4,
#     hdf5_dtype: Optional[Union[np.dtype, str]] = None,
#     hdf5_mutator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
#     # file names
#     file_name: Optional[str] = None,
#     uri_name: Optional[str] = None,
#     # hash settings
#     hash_type: str = 'md5',
#     hash_mode: str = 'fast',
# ) -> DataObject:
#
#     file_name = basename_from_url(uri) if (file_name is None) else file_name
#     uri_name = f'dl.{basename_from_url(uri)}' if (uri_name is None) else uri_name
#
#     @prepare_stalefile(out_name=uri_name, out_hash=uri_hash, hash_type=hash_type, hash_mode=hash_mode)
#     def download_to(out_dir: str, out_path: str):
#         retrieve_file(src_uri=uri, dst_path=out_path, overwrite_existing=True)
#
#     @prepare_stalefile(out_name=file_name, out_hash=file_hash, hash_type=hash_type, hash_mode=hash_mode)
#     def process_to(out_dir: str, out_path: str):
#         inp_path = download_to(out_dir)
#         hdf5_resave_file(
#             inp_path=inp_path,
#             out_path=out_path,
#             dataset_name=hdf5_dataset_name,
#             chunk_size=hdf5_chunk_size,
#             compression=hdf5_compression,
#             compression_lvl=hdf5_compression_lvl,
#             out_dtype=hdf5_dtype,
#             out_mutator=hdf5_mutator,
#         )
#
#     return process_to



# ========================================================================= #
# END                                                                       #
# ========================================================================= #



