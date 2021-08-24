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
import functools
from typing import Optional
from typing import Sequence

import numpy as np


# ========================================================================= #
# Dither Matrix                                                             #
# ========================================================================= #


def nd_dither_offsets(d: int) -> np.ndarray:
    """
    Get the offsets for the d-dimensional dither matrix.

    Output: array of shape: [2]*d

    Algorithm:
    M(d+1) = | 2 *     M(d)      |
             | 2 * flip(M(d)) + 1 |

    Examples:
    d=1:  | d=2:    | d=3:
    ------+---------+----------
    [0 1] | [[0 2]  | [[[0 4]
          |  [3 1]] |   [6 2]]
          |         |  [[3 7]
          |         |   [5 1]]]
    """
    assert isinstance(d, int) and (d > 0)
    # base offsets
    if d == 1:
        return np.array([0, 1])
    # recurse
    prev = nd_dither_offsets(d=d - 1)
    offs = np.array(
        [
            2 * prev,
            2 * np.flip(prev) + 1  # flip(prev) is the same as prev[::-1, >>>] with ::-1 in all dimensions
        ]
    )
    return _np_immutable_copy(offs)


def nd_dither_matrix(n: int = 2, d: int = 2, norm: bool = False) -> np.ndarray:
    """
    Compute the d-dimension dither matrix, with dimensions of size n.
    - n must be a power of 2!

    Output: array of shape: [n]*d

    Algorithm (d=2):
    M(2n) = | 4 * M(n) + 0     4 * M(n) + 2 |
            | 4 * M(n) + 3     4 * M(n) + 1 |

    Examples (n=2, norm=False):
    d=1:  | d=2:    | d=3:
    ------+---------+----------
    [0 1] | [[0 2]  | [[[0 4]
          |  [3 1]] |   [6 2]]
          |         |  [[3 7]
          |         |   [5 1]]]

    Examples (n=4, norm=False):
    d=1:      | d=2:
    ----------+----------------
    [0 2 1 3] | [[ 0  8  2 10]
              |  [12  4 14  6]
              |  [ 3 11  1  9]
              |  [15  7 13  5]]
    """
    assert _is_power_2(n)
    # handle smallest case
    if n == 1:
        return np.zeros([1] * d)  # shape: [1] * d
    # recurse
    offs = nd_dither_offsets(d=d)  # shape: [2] * d
    prev = nd_dither_matrix(n=n // 2, d=d, norm=False)  # shape: [N//2] * d
    # combine
    noffs = np.kron(
        offs, np.ones([n // 2] * d)
    )  # kron, eg. [0, 1] -> [0, 0, 1, 1] | we need to enlarge to shape: [n] * d
    nprev = np.tile(prev, [2] * d)  # tile, eg. [0, 1] -> [0, 1, 0, 1] | we need to enlarge to shape: [n] * d
    next = nprev * offs.size + noffs  # shape: [n] * d
    # return
    if norm:
        next /= next.size
    return _np_immutable_copy(next)  # shape: [n] * d


# ========================================================================= #
# Apply Dithering                                                           #
# ========================================================================= #


def nd_dither(arr: np.ndarray, n: int = 2, axis: Optional[Sequence[int]] = None) -> np.ndarray:
    """
    Apply ordered dithering along the specified axes of an array.
    The array must be floats with values in the range [0, 1]

    If axis is not specified, then all the axes are dithered.

    The output is a boolean array with the same shape as the input arr.
    """
    # convert axis to ordered axis
    src_axis = _normalize_axis(arr.ndim, tuple(axis))
    d = len(axis)
    dst_axis = np.arange(arr.ndim - d, arr.ndim)
    # move the d axes in use to front
    arr = np.moveaxis(arr, src_axis, dst_axis)
    # generate indices
    ii = tuple(ii % n for ii in np.ogrid[tuple(_IDX[0:l] for l in arr.shape[-d:])])  # meshgrid is buggy, doesn't preserve dims correctly
    d_mat = nd_dither_matrix(n=n, d=d, norm=True)
    dd = d_mat[ii]
    # compute
    out = arr > dd
    # return in use axis to original positions
    out = np.moveaxis(out, dst_axis, src_axis)
    # dither
    return out


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def _np_immutable_copy(arr: np.ndarray):
    arr = np.copy(arr)  # does not copy inner python objects
    arr.flags.writeable = False
    return arr


def _is_power_2(num: int):
    assert isinstance(num, int)
    if num <= 0:
        return False
    return not bool(num & (num - 1))


@functools.lru_cache()
def _normalize_axis(ndim: int, axis: Optional[Sequence[int]]) -> np.ndarray:
    # TODO: this functionality may be duplicated
    # defaults
    if axis is None:
        axis = np.arange(ndim)
    # convert
    axis = np.array(axis)
    if axis.ndim == 0:
        axis = axis[None]
    # checks
    assert axis.ndim == 1
    assert axis.dtype in ('int', 'int32', 'int64')
    # convert
    axis = np.where(axis < 0, ndim + axis, axis)
    axis = np.sort(axis)
    # checks
    assert np.unique(axis).shape == axis.shape
    assert np.all(0 <= axis)
    assert np.all(axis < ndim)
    # done!
    return _np_immutable_copy(axis)  # shape: [d]


# hack to enable quick indexing
class __IDX(object):
    def __getitem__(self, item):
        return item


_IDX = __IDX()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
