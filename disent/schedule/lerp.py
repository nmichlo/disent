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


import math
from typing import Union
import numpy as np


# ========================================================================= #
# linear interpolate                                                        #
# ========================================================================= #


def lerp(ratio, start_val, end_val):
    """Linear interpolation between parameters, respects bounds when t is out of bounds [0, 1]"""
    # assert a < b
    r = np.clip(ratio, 0., 1.)
    # precise method, guarantees v==b when t==1 | simplifies to: a + t*(b-a)
    return (1 - r) * start_val + r * end_val
    # return start_val + r * (end_val - start_val)  # EQUIVALENT


def lerp_step(step, max_step, start_val, end_val):
    """Linear interpolation based on a step count."""
    assert max_step > 0
    return lerp(ratio=step / max_step, start_val=start_val, end_val=end_val)


# ========================================================================= #
# linear interpolate                                                        #
# ========================================================================= #


_SCALE_RATIO_FNS = {
    'linear':  lambda r: r,
    'sigmoid': lambda r: 1 / (1 + np.exp(-12 * r + 6)),
    'cosine':  lambda r: 0.5 * (1 - np.cos(r * math.pi)),
}


def scale_ratio(r, mode='linear'):
    r = np.clip(r, 0., 1.)
    return _SCALE_RATIO_FNS[mode](r)


# ========================================================================= #
# Cyclical Annealing Schedules                                              #
# - https://arxiv.org/abs/1903.10145                                        #
# - https://github.com/haofuml/cyclical_annealing                           #
# These functions are not exactly the same, but are more flexible.          #
# ========================================================================= #


_END_VALUES = {
    'low': 0,
    'high': 1,
}


def cyclical_anneal(
    step: Union[int, float, np.ndarray],
    period: float = 3600,
    low_ratio: float = 0.0,
    high_ratio: float = 0.0,
    repeats: int = None,
    start_low: bool = True,
    end_value: str = 'high',
    mode: str = 'linear',
):
    # check values
    assert 0 <= low_ratio <= 1
    assert 0 <= high_ratio <= 1
    assert (low_ratio + high_ratio) <= 1
    assert (period > 0)
    # compute ratio & masks
    r = (step / period) % 1
    # flip the axis
    if not start_low:
        r = 1 - r
    # get the clip mask
    low_mask, high_mask = r <= low_ratio, (1 - high_ratio) <= r
    # compute increasing values
    if low_ratio + high_ratio < 1:
        r = (r - low_ratio) / (1-low_ratio-high_ratio)
        r = scale_ratio(r, mode=mode)
    # truncate values
    r = np.where(low_mask, 0, r)
    r = np.where(high_mask, 1, r)
    # repeats
    if repeats is not None:
        end_value = _END_VALUES.get(end_value, end_value)
        assert 0 <= end_value <= 1
        assert repeats > 0
        # compute
        n = step / period
        r = np.where(n < repeats, r, end_value)
    # done
    return r


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
