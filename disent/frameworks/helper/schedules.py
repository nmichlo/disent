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
# Cyclical Annealing Schedules - Activations                                #
# - same api as the original versions but cleaned up!                       #
# !!! we keep these around to keep the same API as the original functions   #
# ========================================================================= #


def activate_linear(v):  return v
def activate_sigmoid(v): return 1 / (1 + np.exp(-12 * v + 6))
def activate_cosine(v):  return 0.5 * (1 - np.cos(v * math.pi))


_FLERP_ACTIVATIONS = {
    'linear': activate_linear,
    'sigmoid': activate_sigmoid,
    'cosine': activate_cosine,
}


def activate(v, mode='linear'):
    return _FLERP_ACTIVATIONS[mode](v)


# ========================================================================= #
# Cleaned Cyclical Annealing Schedules                                      #
# - same api as the original versions but cleaned up!                       #
# !!! we keep these around to keep the same API as the original functions   #
# ========================================================================= #


def frange_cycle(total_steps, repeats=4, ratio=0.5, mode='linear', v_delta=None):
    L = np.ones(total_steps)
    period = total_steps / repeats
    # check one or the other
    if ratio is not None:
        assert v_delta is None
    if v_delta is not None:
        assert ratio is None
    # handle last case
    if v_delta is None:
        v_delta = 1 / (period * ratio)
    # activate fn
    activation = _FLERP_ACTIVATIONS[mode]
    # linear schedule
    for c in range(repeats):
        v, i = 0, 0
        # this is erroneous... in the original implementation
        # v > v_max resets v to zero, but the remainder v_max - v
        # accumulates and is not used
        while v <= 1 and (int(i + c * period) < total_steps):
            L[int(i + c * period)] = activation(v)
            v += v_delta
            i += 1
    return L


def frange(v_delta, total_steps, mode='linear'):
    return frange_cycle(total_steps, repeats=1, ratio=None, mode=mode, v_delta=v_delta)


# ========================================================================= #
# LERP Cyclical Annealing Schedules                                         #
# - these have better APIs                                                  #
# ========================================================================= #


_END_VALUES = {
    'low': 0,
    'high': 1,
}


def flerp_cycle(
    i: Union[int, float, np.ndarray],
    period: float,
    low_ratio=0.1,
    high_ratio=0.25,
    repeats: int = None,
    start_low=True,
    end_value='high',
    mode='linear',
):
    # check values
    assert 0 <= low_ratio <= 1
    assert 0 <= high_ratio <= 1
    assert (low_ratio + high_ratio) <= 1
    # check values
    assert (period > 0)
    if repeats:
        assert repeats > 0
        end_value = _END_VALUES.get(end_value, end_value)
        assert 0 <= end_value <= 1
    # compute ratio & masks
    r = (i / period) % 1
    # flip the axis
    if not start_low:
        r = 1 - r
    # get the clip mask
    low_mask, high_mask = r <= low_ratio, (1 - high_ratio) <= r
    # compute increasing values
    if low_ratio + high_ratio < 1:
        r = (r - low_ratio) / (1-low_ratio-high_ratio)
        r = activate(r, mode=mode)
    # truncate values
    r = np.where(low_mask, 0, r)
    r = np.where(high_mask, 1, r)
    # repeats
    if repeats is not None:
        n = i / period
        r = np.where(n < repeats, r, end_value)
    # done
    return r

# ========================================================================= #
# END                                                                       #
# ========================================================================= #

