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


# ========================================================================= #
# Deprecate                                                                 #
# ========================================================================= #


def deprecated(msg: str):
    """
    Mark a function or class as deprecated, and print a warning the
    first time it is used.
    - This decorator wraps functions, but only replaces the __init__
      method of a class so that we can still inherit from a deprecated class!
    """
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
                logging.getLogger(name).warning(f'[DEPRECATED] {path} - {repr(dsc)}')
                dat = None
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
