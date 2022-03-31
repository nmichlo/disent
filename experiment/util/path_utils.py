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
import re
from pathlib import Path
from typing import Optional
from typing import Tuple
from typing import Union


log = logging.getLogger(__name__)


# ========================================================================= #
# PATH HELPERS                                                              #
# ========================================================================= #


_EXPERIMENT_SEP = '_'
_EXPERIMENT_RGX = re.compile(f'^([0-9]+)({_EXPERIMENT_SEP}.+)?$')


def get_max_experiment_number(root_dir: str, return_path: bool = False) -> Union[int, Tuple[int, Optional[str]]]:
    """
    Get the next experiment number in the specified directory. Experiment directories
    all start with a numerical value.
    - eg. "1", "00002", "3_name", "00042_name" are all valid subdirectories.
    - eg. "name", "name_1", "name_00001", "99999_image.png" are all invalid and are
          ignored. Either their name format is wrong or they are a file.

    If all the above directories are all used as an example, then this function will
    return the value 42 corresponding to "00042_name"
    """
    # check the dirs exist
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f'The given experiments directory does not exist: {repr(root_dir)} ({repr(os.path.abspath(root_dir))})')
    elif not os.path.isdir(root_dir):
        raise NotADirectoryError(f'The given experiments path exists, but is not a directory: {repr(root_dir)} ({repr(os.path.abspath(root_dir))})')
    # linear search over each file in the dir
    max_num, max_path = 0, None
    for file in os.listdir(root_dir):
        # skip if not a directory
        if not os.path.isdir(os.path.join(root_dir, file)):
            continue
        # skip if the file name does not match
        match = _EXPERIMENT_RGX.search(file)
        if not match:
            continue
        # update the maximum number
        num, _ = match.groups()
        num = int(num)
        if num > max_num:
            max_num, max_path = num, file
    # done!
    if return_path:
        return max_num, max_path
    return max_num


_CURRENT_EXPERIMENT_NUM: Optional[int] = None
_CURRENT_EXPERIMENT_DIR: Optional[str] = None


def get_current_experiment_number(root_dir: str) -> int:
    """
    Get the next experiment number from the experiment directory, and cache
    the result for future calls of this function for the current instance of the program.
    - The next time the program is run, this value will differ.

    For example, if the `root_dir` contains the directories: "00001_name", "00041", then
    this function will return the next value which is `42` on all subsequent calls, even
    if a directory for experiment 42 is created during the current program's lifetime.
    """
    global _CURRENT_EXPERIMENT_NUM
    if _CURRENT_EXPERIMENT_NUM is None:
        _CURRENT_EXPERIMENT_NUM = get_max_experiment_number(root_dir, return_path=False) + 1
    return _CURRENT_EXPERIMENT_NUM


def get_current_experiment_dir(root_dir: str, name: Optional[str] = None) -> str:
    """
    Like `get_current_experiment_number` which computes the next experiment number, this
    function computes the next experiment path, which appends a name to the computed number.

    The result is cached for the lifetime of the program, however, on subsequent calls of
    this function, the computed name must always match the original value otherwise an
    error is thrown! This is to prevent experiments with duplicate numbers from being created!
    """
    if name is not None:
        assert Path(name).name == name, f'The given name is not valid: {repr(name)}'
    # make the dirname & normalise the path
    num = get_current_experiment_number(root_dir)
    dir_name = f'{num:05d}{_EXPERIMENT_SEP}{name}' if name else f'{num:05d}'
    exp_dir = os.path.abspath(os.path.join(root_dir, dir_name))
    # cache the experiment name or check against the existing cache
    global _CURRENT_EXPERIMENT_DIR
    if _CURRENT_EXPERIMENT_DIR is None:
        _CURRENT_EXPERIMENT_DIR = exp_dir
    if exp_dir != _CURRENT_EXPERIMENT_DIR:
        raise RuntimeError(f'Current experiment directory has already been set: {repr(_CURRENT_EXPERIMENT_DIR)} This does not match what was computed: {repr(exp_dir)}')
    # done!
    return _CURRENT_EXPERIMENT_DIR


def make_current_experiment_dir(root_dir: str, name: Optional[str] = None) -> str:
    """
    Like `get_current_experiment_dir`, but create any of the directories if needed.
    - Both the `root_dir` and the computed subdir for the current experiment will be created.
    """
    root_dir = os.path.abspath(root_dir)
    # make the root directory if it does not exist!
    if not os.path.exists(root_dir):
        log.info(f'root experiments directory does not exist, creating... {repr(root_dir)}')
        os.makedirs(root_dir, exist_ok=True)
    # get the current dir
    current_dir = get_current_experiment_dir(root_dir, name)
    # make the current dir
    if not os.path.exists(current_dir):
        log.info(f'current experiment directory does not exist, creating... {repr(current_dir)}')
        os.makedirs(current_dir, exist_ok=True)
    # done!
    return current_dir


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
