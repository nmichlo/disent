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

import inspect
import os
from typing import Optional


# ========================================================================= #
# Files                                                                     #
# ========================================================================= #


def _make_rel_path(*path_segments, is_file=True, _calldepth=0):
    assert not os.path.isabs(os.path.join(*path_segments)), 'path must be relative'
    # get source
    stack = inspect.stack()
    module = inspect.getmodule(stack[_calldepth+1].frame)
    reldir = os.path.dirname(module.__file__)
    # make everything
    path = os.path.join(reldir, *path_segments)
    folder_path = os.path.dirname(path) if is_file else path
    os.makedirs(folder_path, exist_ok=True)
    return path


def _make_rel_path_add_ext(*path_segments, ext='.png', _calldepth=0):
    # make path
    path = _make_rel_path(*path_segments, is_file=True, _calldepth=_calldepth+1)
    if not os.path.splitext(path)[1]:
        path = f'{path}{ext}'
    return path


def make_rel_path(*path_segments, is_file=True):
    return _make_rel_path(*path_segments, is_file=is_file, _calldepth=1)


def make_rel_path_add_ext(*path_segments, ext='.png'):
    return _make_rel_path_add_ext(*path_segments, ext=ext, _calldepth=1)


def plt_rel_path_savefig(rel_path: Optional[str], save: bool = True, show: bool = True, ext='.png', dpi: Optional[int] = None, **kwargs):
    import matplotlib.pyplot as plt
    if save and (rel_path is not None):
        path = _make_rel_path_add_ext(rel_path, ext=ext, _calldepth=2)
        plt.savefig(path, dpi=dpi, **kwargs)
        print(f'saved: {repr(path)}')
    if show:
        plt.show(**kwargs)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
