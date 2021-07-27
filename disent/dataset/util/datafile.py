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

import os
from abc import ABCMeta
from typing import Callable
from typing import Dict
from typing import final
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np

from disent.dataset.util.hdf5 import hdf5_resave_file
from disent.util.inout.cache import stalefile
from disent.util.function import wrapped_partial
from disent.util.inout.files import retrieve_file
from disent.util.inout.paths import filename_from_url
from disent.util.inout.paths import modify_file_name


# ========================================================================= #
# data objects                                                              #
# ========================================================================= #


class DataFile(object, metaclass=ABCMeta):
    """
    base DataFile that does nothing, if the file does
    not exist or it has the incorrect hash, then that's your problem!
    """

    def __init__(self, file_name: str):
        self._file_name = file_name

    @final
    @property
    def out_name(self) -> str:
        return self._file_name

    def prepare(self, out_dir: str) -> str:
        # TODO: maybe check that the file exists or not and raise a FileNotFoundError?
        pass

    def __repr__(self):
        return f'{self.__class__.__name__}(out_name={repr(self.out_name)})'


class DataFileHashed(DataFile, metaclass=ABCMeta):
    """
    Abstract Class
    - Base DataFile class that guarantees a file to exist,
      if the file does not exist, or the hash of the file is
      incorrect, then the file is re-generated.
    """

    def __init__(
        self,
        file_name: str,
        file_hash: Optional[Union[str, Dict[str, str]]],
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        super().__init__(file_name=file_name)
        self._file_hash = file_hash
        self._hash_type = hash_type
        self._hash_mode = hash_mode

    def prepare(self, out_dir: str) -> str:
        @stalefile(file=os.path.join(out_dir, self._file_name), hash=self._file_hash, hash_type=self._hash_type, hash_mode=self._hash_mode)
        def wrapped(out_file):
            self._prepare(out_dir=out_dir, out_file=out_file)
        return wrapped()

    def _prepare(self, out_dir: str, out_file: str) -> str:
        # TODO: maybe raise a FileNotFoundError or a HashError instead?
        raise NotImplementedError


class DataFileHashedDl(DataFileHashed):
    """
    Download a file
    - uri can also be a file to perform a copy instead of download,
      useful for example if you want to retrieve a file from a network drive.
    """

    def __init__(
        self,
        uri: str,
        uri_hash: Optional[Union[str, Dict[str, str]]],
        uri_name: Optional[str] = None,
        hash_type: str = 'md5',
        hash_mode: str = 'fast',
    ):
        super().__init__(
            file_name=filename_from_url(uri) if (uri_name is None) else uri_name,
            file_hash=uri_hash,
            hash_type=hash_type,
            hash_mode=hash_mode
        )
        self._uri = uri

    def _prepare(self, out_dir: str, out_file: str):
        retrieve_file(src_uri=self._uri, dst_path=out_file, overwrite_existing=True)

    def __repr__(self):
        return f'{self.__class__.__name__}(uri={repr(self._uri)}, out_name={repr(self.out_name)})'


class DataFileHashedDlGen(DataFileHashed, metaclass=ABCMeta):
    """
    Abstract class
    - download a file and perform some processing on that file.
    """

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
        self._dl_obj = DataFileHashedDl(
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

    def __repr__(self):
        return f'{self.__class__.__name__}(uri={repr(self._dl_obj._uri)}, uri_name={repr(self._dl_obj.out_name)}, out_name={repr(self.out_name)})'


class DataFileHashedDlH5(DataFileHashedDlGen):
    """
    Downloads an hdf5 file and pre-processes it into the specified chunk_size.
    """

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
        hdf5_obs_shape: Optional[Sequence[int]] = None,
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
            obs_shape=hdf5_obs_shape,
        )
        # save the dataset name
        self._dataset_name = hdf5_dataset_name

    @property
    def dataset_name(self) -> str:
        return self._dataset_name

    def _generate(self, inp_file: str, out_file: str):
        self._hdf5_resave_file(inp_path=inp_file, out_path=out_file)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
