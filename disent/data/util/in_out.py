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
import math
import os
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from disent.util import colors as c


log = logging.getLogger(__name__)


# ========================================================================= #
# Formatting                                                                #
# ========================================================================= #


_BYTES_POW_NAME = ("B  ",  "KiB",  "MiB",  "GiB",  "TiB",  "PiB",  "EiB",  "ZiB",  "YiB")
_BYTES_POW_COLR = (c.WHT, c.lGRN, c.lYLW, c.lRED, c.lRED, c.lRED, c.lRED, c.lRED, c.lRED)


def bytes_to_human(size_bytes, decimals=3, color=True):
    if size_bytes == 0:
        return "0B"
    # round correctly
    i = int(math.floor(math.log(size_bytes, 1024)))
    s = round(size_bytes / math.pow(1024, i), decimals)
    # generate string
    name = f'{_BYTES_POW_COLR[i]}{_BYTES_POW_NAME[i]}{c.RST}' if color else f'{_BYTES_POW_NAME[i]}'
    # format string
    return f"{s:{4+decimals}.{decimals}f} {name}"


# ========================================================================= #
# file hashing                                                              #
# ========================================================================= #


def yield_file_bytes(file: str, chunk_size=16384):
    with open(file, 'rb') as f:
        bytes = True
        while bytes:
            bytes = f.read(chunk_size)
            yield bytes


def yield_fast_hash_bytes(file: str, chunk_size=16384, num_chunks=3):
    assert num_chunks >= 2
    # return the size in bytes
    size = os.path.getsize(file)
    yield size.to_bytes(length=64//8, byteorder='big', signed=False)
    # return file bytes chunks
    if size < chunk_size * num_chunks:
        # we cant return chunks because the file is too small, return everything!
        yield from yield_file_bytes(file, chunk_size=chunk_size)
    else:
        # includes evenly spaced start, middle and end chunks
        with open(file, 'rb') as f:
            for i in range(num_chunks):
                pos = (i * (size - chunk_size)) // (num_chunks - 1)
                f.seek(pos)
                yield f.read(chunk_size)


def hash_file(file: str, hash_type='md5', hash_mode='full') -> str:
    """
    :param file: the path to the file
    :param hash_type: the kind of hash to compute, default is "md5"
    :param hash_mode: "full" uses all the bytes in the file to compute the hash, "fast" uses the start, middle, end bytes as well as the size of the file in the hash.
    :param chunk_size: number of bytes to read at a time
    :return: the hexdigest of the hash
    """
    import hashlib
    # get file bytes iterator
    if hash_mode == 'full':
        byte_iter = yield_file_bytes(file=file)
    elif hash_mode == 'fast':
        byte_iter = yield_fast_hash_bytes(file=file)
    else:
        raise KeyError(f'invalid hash_mode: {repr(hash_mode)}')
    # generate hash
    hash = hashlib.new(hash_type)
    for bytes in byte_iter:
        hash.update(bytes)
    hash = hash.hexdigest()
    # done
    return hash


class HashError(Exception):
    """
    Raised if the hash of a file was invalid.
    """


def validate_file_hash(file: str, hash: Union[str, Dict[str, str]], hash_type='md5', hash_mode='full'):
    if isinstance(hash, dict):
        hash = hash[hash_mode]
    fhash = hash_file(file=file, hash_type=hash_type, hash_mode=hash_mode)
    if fhash != hash:
        msg = f'computed {hash_mode} {hash_type} hash: {repr(fhash)} does not match expected hash: {repr(hash)} for file: {repr(file)}'
        log.error(msg)
        raise HashError(msg)


# ========================================================================= #
# Atomic file saving                                                        #
# ========================================================================= #


class AtomicFileContext(object):
    """
    Within the context, data must be written to a temporary file.
    Once data has been successfully written, the temporary file
    is moved to the location of the given file.

    ```
    with AtomicFileHandler('file.txt') as tmp_file:
        with open(tmp_file, 'w') as f:
            f.write("hello world!\n")
    ```

    # TODO: can this be cleaned up with the TemporaryDirectory and TemporaryFile classes?
    """

    def __init__(
        self,
        file: str,
        open_mode: Optional[str] = None,
        overwrite: bool = False,
        makedirs: bool = True,
        tmp_file: Optional[str] = None,
        tmp_prefix: str = '_TEMP_.',
        tmp_postfix: str = '',
    ):
        from pathlib import Path
        # check files
        if not file:
            raise ValueError(f'file must not be empty: {repr(file)}')
        if not tmp_file and (tmp_file is not None):
            raise ValueError(f'tmp_file must not be empty: {repr(tmp_file)}')
        # get files
        self.trg_file = Path(file).absolute()
        tmp_file = Path(self.trg_file if (tmp_file is None) else tmp_file)
        self.tmp_file = tmp_file.parent.joinpath(f'{tmp_prefix}{tmp_file.name}{tmp_postfix}')
        # check that the files are different
        if self.trg_file == self.tmp_file:
            raise ValueError(f'temporary and target files are the same: {self.tmp_file} == {self.trg_file}')
        # other settings
        self._makedirs = makedirs
        self._overwrite = overwrite
        self._open_mode = open_mode
        self._resource = None

    def __enter__(self):
        # check files exist or not
        if self.tmp_file.exists():
            if not self.tmp_file.is_file():
                raise FileExistsError(f'the temporary file exists but is not a file: {self.tmp_file}')
        if self.trg_file.exists():
            if not self._overwrite:
                raise FileExistsError(f'the target file already exists: {self.trg_file}, set overwrite=True to ignore this error.')
            if not self.trg_file.is_file():
                raise FileExistsError(f'the target file exists but is not a file: {self.trg_file}')
        # create the missing directories if needed
        if self._makedirs:
            self.tmp_file.parent.mkdir(parents=True, exist_ok=True)
        # delete any existing temporary files
        if self.tmp_file.exists():
            log.debug(f'deleting existing temporary file: {self.tmp_file}')
            self.tmp_file.unlink()
        # handle the different modes, deleting any existing tmp files
        if self._open_mode is not None:
            log.debug(f'created new temporary file: {self.tmp_file}')
            self._resource = open(self.tmp_file, self._open_mode)
            return str(self.tmp_file), self._resource
        else:
            return str(self.tmp_file)

    def __exit__(self, error_type, error, traceback):
        # close the temp file
        if self._resource is not None:
            self._resource.close()
            self._resource = None
        # cleanup if there was an error, and exit early
        if error_type is not None:
            if self.tmp_file.exists():
                self.tmp_file.unlink(missing_ok=True)
                log.error(f'An error occurred in {self.__class__.__name__}, cleaned up temporary file: {self.tmp_file}')
            else:
                log.error(f'An error occurred in {self.__class__.__name__}')
            return
        # the temp file must have been created!
        if not self.tmp_file.exists():
            raise FileNotFoundError(f'the temporary file was not created: {self.tmp_file}')
        # delete the target file if it exists and overwrite is enabled:
        if self._overwrite:
            log.warning(f'overwriting file: {self.trg_file}')
            self.trg_file.unlink(missing_ok=True)
        # create the missing directories if needed
        if self._makedirs:
            self.trg_file.parent.mkdir(parents=True, exist_ok=True)
        # move the temp file to the target file
        log.info(f'moved temporary file to final location: {self.tmp_file} -> {self.trg_file}')
        os.rename(self.tmp_file, self.trg_file)


# ========================================================================= #
# files/dirs exist                                                          #
# ========================================================================= #


def ensure_dir_exists(*path, is_file=False, absolute=False):
    import os
    # join path
    path = os.path.join(*path)
    # to abs path
    if absolute:
        path = os.path.abspath(path)
    # remove file
    dirs = os.path.dirname(path) if is_file else path
    # create missing directory
    if os.path.exists(dirs):
        if not os.path.isdir(dirs):
            raise IOError(f'path is not a directory: {dirs}')
    else:
        os.makedirs(dirs, exist_ok=True)
        log.info(f'created missing directories: {dirs}')
    # return directory
    return path


def ensure_parent_dir_exists(*path):
    return ensure_dir_exists(*path, is_file=True, absolute=True)


# ========================================================================= #
# files/dirs exist                                                          #
# ========================================================================= #


def download_file(url: str, save_path: str, overwrite_existing: bool = False, chunk_size: int = 16384):
    import requests
    from tqdm import tqdm
    # write the file
    with AtomicFileContext(file=save_path, open_mode='wb', overwrite=overwrite_existing) as (_, file):
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')
        # cast to integer if content-length exists on response
        if total_length is not None:
            total_length = int(total_length)
        # download with progress bar
        log.info(f'Downloading: {url} to: {save_path}')
        with tqdm(total=total_length, desc=f'Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
            for data in response.iter_content(chunk_size=chunk_size):
                file.write(data)
                progress.update(chunk_size)


def copy_file(src: str, dst: str, overwrite_existing: bool = False):
    # copy the file
    if os.path.abspath(src) == os.path.abspath(dst):
        raise FileExistsError(f'input and output paths for copy are the same, skipping: {repr(dst)}')
    else:
        with AtomicFileContext(file=dst, overwrite=overwrite_existing) as path:
            import shutil
            shutil.copyfile(src, path)


def retrieve_file(src_uri: str, dst_path: str, overwrite_existing: bool = True):
    uri, is_url = _uri_parse_file_or_url(src_uri)
    if is_url:
        download_file(url=uri, save_path=dst_path, overwrite_existing=overwrite_existing)
    else:
        copy_file(src=uri, dst=dst_path, overwrite_existing=overwrite_existing)


# ========================================================================= #
# path utils                                                                #
# ========================================================================= #


def basename_from_url(url):
    import os
    from urllib.parse import urlparse
    return os.path.basename(urlparse(url).path)


def _uri_parse_file_or_url(inp_uri) -> Tuple[str, bool]:
    from urllib.parse import urlparse
    result = urlparse(inp_uri)
    # parse different cases
    if result.scheme in ('http', 'https'):
        is_url = True
        uri = result.geturl()
    elif result.scheme in ('file', ''):
        is_url = False
        if result.scheme == 'file':
            if result.netloc:
                raise KeyError(f'file uri format is invalid: "{result.geturl()}" two slashes specifies host as: "{result.netloc}" eg. instead of "file://hostname/root_folder/file.txt", please use: "file:/root_folder/file.txt" (no hostname) or "file:///root_folder/file.txt" (empty hostname).')
            if not os.path.isabs(result.path):
                raise RuntimeError(f'path: {repr(result.path)} obtained from file URI: {repr(inp_uri)} should always be absolute')
            uri = result.path
        else:
            uri = result.geturl()
        uri = os.path.abspath(uri)
    else:
        raise ValueError(f'invalid file or url: {repr(inp_uri)}')
    # done
    return uri, is_url


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
