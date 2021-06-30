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

from disent.dataset.util.state_space import StateSpace


# ========================================================================= #
# TEST STATE SPACE                                                          #
# ========================================================================= #


FACTOR_SIZES = [
    [5, 4, 3, 2],
    [2, 3, 4, 5],
    [2, 3, 4],
    [1, 2, 3],
    [1, 33, 1],
    [1, 1, 1],
    [1],
]

def test_discrete_state_space_single_values():
    for factor_sizes in FACTOR_SIZES:
        states = StateSpace(factor_sizes=factor_sizes)
        # check size
        assert len(states) == np.prod(factor_sizes)
        # check single values
        for i, f in enumerate(states.factor_sizes):
            factors = states.factor_sizes - 1
            factors[:i] = 0
            idx = np.prod(states.factor_sizes[i:]) - 1
            assert states.pos_to_idx(factors) == idx
            assert np.all(states.idx_to_pos(idx) == factors)

def test_discrete_state_space_one_to_one():
    for factor_sizes in FACTOR_SIZES:
        states = StateSpace(factor_sizes=factor_sizes)
        # check that entire range of values is generated
        k = np.random.randint(1, 5)
        # chances of this failing are extremely low, but it could happen...
        pos_0 = states.sample_factors([int(100_000 ** (1/k))] * k)
        # check random values are in the right ranges
        all_dims = tuple(range(pos_0.ndim))
        assert np.all(np.max(pos_0, axis=all_dims[:-1]) == (states.factor_sizes - 1))
        assert np.all(np.min(pos_0, axis=all_dims[:-1]) == 0)
        # check that converting between them keeps values the same
        idx_0 = states.pos_to_idx(pos_0)
        pos_1 = states.idx_to_pos(idx_0)
        idx_1 = states.pos_to_idx(pos_1)
        assert np.all(idx_0 == idx_1)
        assert np.all(pos_0 == pos_1)


def test_new_functions():
    # TODO: convert to propper tests
    s = StateSpace([2, 4, 6])
    maxs = np.max([s.sample_factors((2, 2), factor_indices=[2, 1, 2, 2]) for i in range(100)], axis=0)
    maxs = np.max([s.sample_missing_factors([[1, 1], [2, 2]], known_factor_indices=[0, 2]) for i in range(100)], axis=0)
    # print(np.min([s.resample_radius([[0, 1, 2], [0, 0, 0]], resample_radius=1, distinct=True) for i in range(1000)], axis=0).tolist())
    # print(np.max([s.resample_radius([[0, 1, 2], [0, 0, 0]], resample_radius=1, distinct=True) for i in range(1000)], axis=0).tolist())


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
