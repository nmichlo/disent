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


import numpy as np


# ========================================================================= #
# Better Choice                                                             #
# ========================================================================= #


def random_choice_prng(a, size=None, replace=True, seed: int = None):
    # generate a random seed
    if seed is None:
        seed = np.random.randint(0, 2**32)
    # create seeded pseudo random number generator
    # - built in np.random.choice cannot handle large values: https://github.com/numpy/numpy/issues/5299#issuecomment-497915672
    # - PCG64 is the default: https://numpy.org/doc/stable/reference/random/bit_generators/index.html
    # - PCG64 has good statistical properties and is fast: https://numpy.org/doc/stable/reference/random/performance.html
    g = np.random.Generator(np.random.PCG64(seed=seed))
    # sample indices
    choices = g.choice(a, size=size, replace=replace)
    # done!
    return choices


# ========================================================================= #
# Random Ranges                                                             #
# ========================================================================= #


def randint2(a_low, a_high, b_low, b_high, size=None):
    """
    Like np.random.randint, but supports two ranges of values.
    Samples with equal probability from both ranges.
    - a: [a_low, a_high) -> including a_low, excluding a_high!
    - b: [b_low, b_high) -> including b_low, excluding b_high!
    """
    # convert
    a_low, a_high = np.array(a_low), np.array(a_high)
    b_low, b_high = np.array(b_low), np.array(b_high)
    # checks
    assert np.all(a_low <= a_high), f'a_low <= a_high | {a_low} <= {a_high}'
    assert np.all(b_low <= b_high), f'b_low <= b_high | {b_low} <= {b_high}'
    assert np.all(a_high <= b_low), f'a_high <= b_low | {a_high} <= {b_low}'
    # compute
    da = a_high - a_low
    db = b_high - b_low
    d = da + db
    assert np.all(d > 0), f'(a_high - a_low) + (b_high - b_low) > 0 | {d} = ({a_high} - {a_low}) + ({b_high} - {b_low}) > 0'
    # sampled
    offset = np.random.randint(0, d, size=size)
    offset += (da <= offset) * (b_low - a_high)
    return a_low + offset


def sample_radius(value, low, high, r_low, r_high):
    """
    Sample around the given value (low <= value < high),
    the resampled value will lie in th same range.
    - sampling occurs in a radius around the value
      r_low <= radius < r_high
    """
    value = np.array(value)
    assert np.all(low <= value)
    assert np.all(value < high)
    # sample for new value
    return randint2(
        a_low=np.maximum(value - r_high + 1, low),
        a_high=np.maximum(value - r_low + 1, low),
        # if r_min == 0, then the ranges overlap, so we must shift one of them.
        b_low=np.minimum(value + r_low + (r_low == 0), high),
        b_high=np.minimum(value + r_high, high),
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
