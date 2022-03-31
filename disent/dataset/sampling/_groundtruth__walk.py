#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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

from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple

import numpy as np

from disent.dataset.data import GroundTruthData
from disent.dataset.sampling import BaseDisentSampler
from disent.dataset.util.state_space import StateSpace
from disent.util.jit import try_njit


# ========================================================================= #
# Pretend We Are Walking Ground-Truth Factors Randomly                      #
# ========================================================================= #


class GroundTruthRandomWalkSampler(BaseDisentSampler):

    def uninit_copy(self) -> 'GroundTruthRandomWalkSampler':
        return GroundTruthRandomWalkSampler(
            num_samples=self._num_samples,
            p_dist_max=self._p_dist_max,
            n_dist_max=self._n_dist_max,
        )

    def __init__(
        self,
        num_samples: int = 3,
        p_dist_max: int = 8,
        n_dist_max: int = 32,
    ):
        super().__init__(num_samples=num_samples)
        # checks
        assert num_samples in {1, 2, 3}, f'num_samples ({repr(num_samples)}) must be 1, 2 or 3'
        # save hparams
        self._num_samples = num_samples
        self._p_dist_max = p_dist_max
        self._n_dist_max = n_dist_max
        # dataset variable
        self._state_space: Optional[StateSpace] = None

    def _init(self, dataset: GroundTruthData):
        assert isinstance(dataset, GroundTruthData), f'dataset must be an instance of {repr(GroundTruthData.__class__.__name__)}, got: {repr(dataset)}'
        self._state_space = dataset.state_space_copy()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _sample_idx(self, idx) -> Tuple[int, ...]:
        if self._num_samples == 1:
            return (idx,)
        elif self._num_samples == 2:
            p_dist = np.random.randint(1, self._p_dist_max + 1)
            pos = _random_walk(idx, p_dist, self._state_space.factor_sizes)
            return (idx, pos)
        elif self._num_samples == 3:
            p_dist = np.random.randint(1, self._p_dist_max + 1)
            n_dist = np.random.randint(1, self._n_dist_max + 1)
            pos = _random_walk(idx, p_dist, self._state_space.factor_sizes)
            neg = _random_walk(pos, n_dist, self._state_space.factor_sizes)
            return (idx, pos, neg)
        else:
            raise RuntimeError


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


def _random_walk(idx: int, dist: int, factor_sizes: np.ndarray) -> int:
    # random walk
    pos = np.array(np.unravel_index(idx, factor_sizes), dtype=int)  # much faster than StateSpace.idx_to_pos, we don't need checks!
    for _ in range(dist):
        _walk_nearby_inplace(pos, factor_sizes)
    idx = np.ravel_multi_index(pos, factor_sizes)  # much faster than StateSpace.pos_to_idx, we don't need checks!
    # done!
    return int(idx)


@try_njit()
def _walk_nearby_inplace(pos: np.ndarray, factor_sizes: Sequence[int]) -> NoReturn:
    # try to shift any single factor by 1 or -1
    while True:
        f_idx = np.random.randint(0, len(factor_sizes))
        cur = pos[f_idx]
        # walk random factor value
        if np.random.random() < 0.5:
            nxt = max(cur - 1, 0)
        else:
            nxt = min(cur + 1, factor_sizes[f_idx] - 1)
        # exit if different
        if cur != nxt:
            break
    # update the position
    pos[f_idx] = nxt


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
