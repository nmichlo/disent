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


from typing import Tuple


# ========================================================================= #
# Import Helper                                                             #
# ========================================================================= #


def _check_and_split_path(import_path: str) -> Tuple[str, ...]:
    segments = import_path.split('.')
    # make sure each segment is a valid python identifier
    if not all(map(str.isidentifier, segments)):
        raise ValueError(f'import path is invalid: {repr(import_path)}')
    # return the segments!
    return tuple(segments)


def import_obj(import_path: str):
    # checks
    segments = _check_and_split_path(import_path)
    # split path
    module_path, attr_name = '.'.join(segments[:-1]), segments[-1]
    # import the module
    import importlib
    try:
        module = importlib.import_module(module_path)
    except Exception as e:
        raise ImportError(f'failed to import module: {repr(module_path)}') from e
    # import the attrs
    try:
        attr = getattr(module, attr_name)
    except Exception as e:
        raise ImportError(f'failed to get attribute: {repr(attr_name)} on module: {repr(module_path)}') from e
    # done
    return attr


def import_obj_partial(import_path: str, *partial_args, **partial_kwargs):
    obj = import_obj(import_path)
    # wrap the object if needed
    if partial_args or partial_kwargs:
        from disent.util.function import wrapped_partial
        obj = wrapped_partial(obj, *partial_args, **partial_kwargs)
    # done!
    return obj


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
