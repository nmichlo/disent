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
from typing import Tuple
from typing import Union


log = logging.getLogger(__name__)


# ========================================================================= #
# Formatting                                                                #
# ========================================================================= #


def modify_file_name(file: Union[str, Path], prefix: str = None, suffix: str = None, sep='.') -> Union[str, Path]:
    # get path components
    path = Path(file)
    assert path.name, f'file name cannot be empty: {repr(path)}, for name: {repr(path.name)}'
    # create new path
    prefix = '' if (prefix is None) else f'{prefix}{sep}'
    suffix = '' if (suffix is None) else f'{sep}{suffix}'
    new_path = path.parent.joinpath(f'{prefix}{path.name}{suffix}')
    # return path
    return str(new_path) if isinstance(file, str) else new_path


# ========================================================================= #
# files/dirs exist                                                          #
# ========================================================================= #


def ensure_dir_exists(*join_paths: str, is_file=False, absolute=False):
    import os
    # join path
    path = os.path.join(*join_paths)
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


def ensure_parent_dir_exists(*join_paths: str):
    return ensure_dir_exists(*join_paths, is_file=True, absolute=True)


# ========================================================================= #
# URI utils                                                                 #
# ========================================================================= #


def filename_from_url(url: str):
    import os
    from urllib.parse import urlparse
    return os.path.basename(urlparse(url).path)


def uri_parse_file_or_url(inp_uri: str) -> Tuple[str, bool]:
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
