from disent.dataset.util.state_space import StateSpace
import numpy as np

FACTOR_SIZES = [
    [5, 4, 3, 2],
    [2, 3, 4, 5],
    [2, 3, 4],
    [1, 2, 3],
    [1, 100, 1],
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
        # chances of this failing are extremely low, but it could happen...
        pos_0 = states.sample_factors(100_000)
        assert np.all(pos_0.max(axis=0) == (states.factor_sizes - 1))
        assert np.all(pos_0.min(axis=0) == 0)
        # check that converting between them keeps values the same
        idx_0 = states.pos_to_idx(pos_0)
        pos_1 = states.idx_to_pos(idx_0)
        idx_1 = states.pos_to_idx(pos_1)
        assert np.all(idx_0 == idx_1)
        assert np.all(pos_0 == pos_1)