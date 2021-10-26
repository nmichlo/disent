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
from pathlib import Path
from typing import Optional
from typing import Union
from uuid import uuid4

from disent.util.inout.paths import uri_parse_file_or_url
from disent.util.inout.paths import modify_file_name


log = logging.getLogger(__name__)


# ========================================================================= #
# Atomic file saving                                                        #
# ========================================================================= #


class AtomicSaveFile(object):
    """
    Within the context, data must be written to a temporary file.
    Once data has been successfully written, the temporary file
    is moved to the location of the target file.

    The temporary file is created in the same directory as the target file.

    ```
    with AtomicFileHandler('file.txt') as tmp_file:
        with open(tmp_file, 'w') as f:
            f.write("hello world!\n")
    ```

    # TODO: can this be cleaned up with the TemporaryDirectory and TemporaryFile classes?
    """

    def __init__(
        self,
        file: Union[str, Path],
        open_mode: Optional[str] = None,
        overwrite: bool = False,
        makedirs: bool = True,
        tmp_prefix: Optional[str] = '.temp.',
        tmp_suffix: Optional[str] = None,
    ):
        # check files
        if not file or not Path(file).name:
            raise ValueError(f'file must not be empty: {repr(file)}')
        # get files
        self.trg_file = Path(file).absolute()
        self.tmp_file = modify_file_name(self.trg_file, prefix=f'{tmp_prefix}{uuid4()}', suffix=tmp_suffix)
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
            if self.trg_file.exists():
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


def download_file(url: str, save_path: str, overwrite_existing: bool = False, chunk_size: int = 16384):
    import requests
    from tqdm import tqdm
    # write the file
    with AtomicSaveFile(file=save_path, open_mode='wb', overwrite=overwrite_existing) as (_, file):
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
        with AtomicSaveFile(file=dst, overwrite=overwrite_existing) as path:
            import shutil
            shutil.copyfile(src, path)


def retrieve_file(src_uri: str, dst_path: str, overwrite_existing: bool = False):
    uri, is_url = uri_parse_file_or_url(src_uri)
    if is_url:
        download_file(url=uri, save_path=dst_path, overwrite_existing=overwrite_existing)
    else:
        copy_file(src=uri, dst=dst_path, overwrite_existing=overwrite_existing)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
