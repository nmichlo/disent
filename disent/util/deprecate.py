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
from functools import wraps
from typing import Optional


# ========================================================================= #
# Deprecate                                                                 #
# ========================================================================= #


def _get_traceback_string() -> str:
    from io import StringIO
    import traceback
    # print the stack trace to an in-memory buffer
    file = StringIO()
    traceback.print_stack(file=file)
    return file.getvalue()


def _get_traceback_file_groups():
    # filter the lines
    results = []
    group = []
    for line in _get_traceback_string().splitlines():
        if line.strip().startswith('File "'):
            if group:
                results.append(group)
                group = []
        group.append(line)
    if group:
        results.append(group)
    return results


def _get_stack_file_strings():
    # mimic the output of a traceback so pycharm performs syntax highlighting when printed
    import inspect
    results = []
    for frame_info in inspect.stack():
        results.append(f'File "{frame_info.filename}", line {frame_info.lineno}, in {frame_info.function}')
    return results[::-1]


_TRACEBACK_MODES = {'none', 'first', 'mini', 'traceback'}
DEFAULT_TRACEBACK_MODE = 'first'


def deprecated(msg: str, traceback_mode: Optional[str] = None):
    """
    Mark a function or class as deprecated, and print a warning the
    first time it is used.
    - This decorator wraps functions, but only replaces the __init__
      method of a class so that we can still inherit from a deprecated class!
    """
    assert isinstance(msg, str), f'msg must be a str, got type: {type(msg)}'
    if traceback_mode is None:
        traceback_mode = DEFAULT_TRACEBACK_MODE
    assert traceback_mode in _TRACEBACK_MODES, f'invalid traceback_mode, got: {repr(traceback_mode)}, must be one of: {sorted(_TRACEBACK_MODES)}'

    def _decorator(fn):
        # we need to handle classes and function separately
        is_class = isinstance(fn, type) and hasattr(fn, '__init__')
        # backup the original function & data
        call_fn = fn.__init__ if is_class else fn
        dat = (fn.__module__, f'{fn.__module__}.{fn.__name__}', str(msg))
        # wrapper function
        @wraps(call_fn)
        def _caller(*args, **kwargs):
            nonlocal dat
            # print the message!
            if dat is not None:
                name, path, dsc = dat
                logger = logging.getLogger(name)
                logger.warning(f'[DEPRECATED] {path} - {repr(dsc)}')
                # get stack trace lines
                if traceback_mode == 'first': lines = _get_stack_file_strings()[-3:-2]
                elif traceback_mode == 'mini': lines = _get_stack_file_strings()[:-2]
                elif traceback_mode == 'traceback': lines = (l[2:] for g in _get_traceback_file_groups()[:-3] for l in g)
                else: lines = []
                # print lines
                for line in lines:
                    logger.warning(f'| {line}')
                # never run this again
                dat = None
            # call the function
            return call_fn(*args, **kwargs)
        # handle function or class
        if is_class:
            fn.__init__ = _caller
        else:
            fn = _caller
        return fn
    return _decorator


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
