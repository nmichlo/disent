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

from itertools import islice
from typing import List
from typing import Sequence


# ========================================================================= #
# Iterators                                                                 #
# ========================================================================= #


def chunked(arr, chunk_size: int, include_remainder=True):
    """
    return an array of array chucks of size chunk_size.
    This is NOT an iterable, and returns all the data.
    """
    size = (len(arr) + chunk_size - 1) if include_remainder else len(arr)
    return [arr[chunk_size*i:chunk_size*(i+1)] for i in range(size // chunk_size)]


def iter_chunks(items, chunk_size: int, include_remainder=True):
    """
    iterable version of chunked.
    that does not evaluate unneeded elements
    """
    items = iter(items)
    for first in items:
        chunk = [first, *islice(items, chunk_size-1)]
        if len(chunk) >= chunk_size or include_remainder:
            yield chunk


def iter_rechunk(chunks, chunk_size: int, include_remainder=True):
    """
    takes in chunks and returns chunks of a new size.
    - Does not evaluate unneeded chunks
    """
    return iter_chunks(
        (item for chunk in chunks for item in chunk),  # flatten chunks
        chunk_size=chunk_size,
        include_remainder=include_remainder
    )


def map_all(fn, *arg_lists, starmap: bool = True, collect_returned: bool = False, common_kwargs: dict = None):
    # TODO: not actually an iterator
    assert arg_lists, 'an empty list of args was passed'
    # check all lengths are the same
    num = len(arg_lists[0])
    assert num > 0
    assert all(len(items) == num for items in arg_lists)
    # update kwargs
    if common_kwargs is None:
        common_kwargs = {}
    # map everything
    if starmap:
        results = (fn(*args, **common_kwargs) for args in zip(*arg_lists))
    else:
        results = (fn(args, **common_kwargs) for args in zip(*arg_lists))
    # zip everything
    if collect_returned:
        return tuple(zip(*results))
    else:
        return tuple(results)


def collect_dicts(results: List[dict]):
    # collect everything
    keys = results[0].keys()
    values = zip(*([result[k] for k in keys] for result in results))
    return {k: list(v) for k, v in zip(keys, values)}


def aggregate_dict(results: dict, reduction='mean'):
    # TODO: this shouldn't be here
    assert reduction == 'mean', 'mean is the only mode supported'
    return {
        k: sum(v) / len(v) for k, v in results.items()
    }


# ========================================================================= #
# Base Class                                                                #
# ========================================================================= #


class LengthIter(Sequence):

    def __iter__(self):
        # this takes priority over __getitem__, otherwise __getitem__ would need to
        # raise an IndexError if out of bounds to signal the end of iteration
        yield from (self[i] for i in range(len(self)))

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, item):
        raise NotImplementedError()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
