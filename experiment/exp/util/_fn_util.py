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
from typing import Sequence


# ========================================================================= #
# Function Arguments                                                        #
# ========================================================================= #


def _get_fn_from_stack(fn_name: str, stack):
    # -- do we actually need all of this?
    fn = None
    for s in stack:
        if fn_name in s.frame.f_locals:
            fn = s.frame.f_locals[fn_name]
            break
    if fn is None:
        raise RuntimeError(f'could not retrieve function: {repr(fn_name)} from call stack.')
    return fn


def get_caller_params(sort: bool = False, exclude: Sequence[str] = None) -> dict:
    stack = inspect.stack()
    fn_name = stack[1].function
    fn_locals = stack[1].frame.f_locals
    # get function and params
    fn = _get_fn_from_stack(fn_name, stack)
    fn_params = inspect.getfullargspec(fn).args
    # check excluded
    exclude = set() if (exclude is None) else set(exclude)
    fn_params = [p for p in fn_params if (p not in exclude)]
    # sort values
    if sort:
        fn_params = sorted(fn_params)
    # return dict
    return {
        k: fn_locals[k] for k in fn_params
    }


def params_as_string(params: dict, sep: str = '_', names: bool = False):
    # get strings
    if names:
        return sep.join(f"{k}={v}" for k, v in params.items())
    else:
        return sep.join(f"{v}" for k, v in params.items())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
