import numpy as np


# ========================================================================= #
# Abstract/Base State Space                                                 #
# ========================================================================= #


class _BaseStateSpace(object):
    @property
    def size(self):
        """The number of permutations of factors handled by this state space"""
        raise NotImplementedError
    @property
    def num_factors(self):
        """The number of factors handled by this state space"""
        raise NotImplementedError
    @property
    def factor_sizes(self):
        """A list of sizes or dimensionality of factors handled by this state space"""
        raise NotImplementedError
    
    def __len__(self):
        """Same as self.size"""
        return self.size
    def __getitem__(self, idx):
        """same as self.idx_to_pos"""
        return self.idx_to_pos(idx)
    def __iter__(self):
        """iterate over all indices and return a corresponding coordinate/position vector"""
        for idx in range(self.size):
            yield self[idx]
    
    def pos_to_idx(self, positions):
        """
        Convert a position to an index (or convert a list of positions to a list of indices)
        - positions are lists of integers, with each element < their corresponding factor size
        - indices are integers < size
        """
        raise NotImplementedError
    def idx_to_pos(self, indices):
        """
        Convert an index to a position (or convert a list of indices to a list of positions)
        - indices are integers < size
        - positions are lists of integers, with each element < their corresponding factor size
        """
        raise NotImplementedError
    
    def sample_factors(self, num_samples, factor_indices=None):
        """
        sample randomly from all factors, otherwise the given factor_indices.
        returned values must appear in the same order as factor_indices.
        """
        raise NotImplementedError
    def sample_missing_factors(self, partial_factors, partial_factor_indices):
        """
        Samples the remaining factors not given in the partial_factor_indices.
        ie. fills in the missing values by sampling from the unused dimensions.
        returned values are ordered by increasing factor index and not factor_indices.
        (partial_factors must correspond to partial_factor_indices)
        """
        raise NotImplementedError
    def resample_factors(self, factors, fixed_factor_indices):
        """
        Resample across all the factors, keeping factor_indices constant.
        returned values are ordered by increasing factor index and not factor_indices.
        """
        raise NotImplementedError


# ========================================================================= #
# Basic State Space                                                         #
# ========================================================================= #


class StateSpace(_BaseStateSpace):
    """
    State space where an index corresponds to coordinates (factors/positions) in the factor space.
    ie. State space with multiple factors of variation, where each factor can be a different size.
    """

    def __init__(self, factor_sizes):
        super().__init__()
        # dimension
        self._factor_sizes = np.array(factor_sizes)
        self._factor_sizes.flags.writeable = False
        # total permutations
        self._size = int(np.prod(factor_sizes))
        # dimension sampling
        self._factor_indices_set = set(range(self.num_factors))

    @property
    def size(self):
        return self._size

    @property
    def num_factors(self):
        return len(self._factor_sizes)

    @property
    def factor_sizes(self):
        return self._factor_sizes
    
    def pos_to_idx(self, positions):
        positions = np.array(positions).T
        return np.ravel_multi_index(positions, self._factor_sizes)

    def idx_to_pos(self, indices):
        positions = np.unravel_index(indices, self._factor_sizes)
        return np.array(positions).T

    def sample_factors(self, num_samples, factor_indices=None):
        sizes = self._factor_sizes if (factor_indices is None) else self._factor_sizes[factor_indices]
        return np.random.randint(sizes, size=(num_samples, len(sizes)))

    def sample_missing_factors(self, known_factors, known_factor_indices):
        num_samples, num_dims = known_factors.shape
        used_indices_set = set(known_factor_indices)
        # assertions
        assert num_dims == len(known_factor_indices), 'dimension count mismatch'
        assert len(used_indices_set) == len(known_factor_indices), 'factor indices are repeated'
        # set used dimensions
        all_values = np.zeros(shape=(num_samples, self.num_factors), dtype=np.int64)
        all_values[:, known_factor_indices] = known_factors
        # sample for missing
        missing_indices = list(self._factor_indices_set - used_indices_set)
        all_values[:, missing_indices] = self.sample_factors(num_samples=num_samples, factor_indices=missing_indices)
        # return
        return all_values

    def resample_factors(self, factors, fixed_factor_indices):
        return self.sample_missing_factors(factors[:, fixed_factor_indices], fixed_factor_indices)


# ========================================================================= #
# Hidden State Space                                                        #
# ========================================================================= #


class HiddenStateSpace(_BaseStateSpace):
    
    """
    State space where an index corresponds to coordinates (factors/positions) in the factor space.
    HOWEVER: some factors are treated as hidden/unknown and are thus randomly sampled.
    
    Inputs to functions act as if known_factor_indices is the new factor space (including factor indexes).
    Outputs from functions act in the full factor space.
    """
    
    def __init__(self, factor_sizes, known_factor_indices):
        factor_sizes, known_factor_indices = np.array(factor_sizes), np.array(known_factor_indices)
        # known factors indices
        self._known_factor_indices = known_factor_indices
        self._known_factor_indices.flags.writeable = False
        self._known_factor_sizes = factor_sizes[known_factor_indices]
        # known state space does not include hidden variables, and assumes they are randomly sampled
        self._known_state_space = StateSpace(self._known_factor_sizes)
        # full state space includes hidden variables
        self._full_state_space = StateSpace(factor_sizes)

    @property
    def size(self):
        return self._known_state_space.size
    
    @property
    def num_factors(self):
        return self._known_state_space.num_factors
    
    @property
    def factor_sizes(self):
        return self._known_state_space.factor_sizes
    
    def pos_to_idx(self, positions):
        positions = np.array(positions)
        if positions.ndim == 1:
            return self._known_state_space.pos_to_idx(positions[self._known_factor_indices])
        elif positions.ndim == 2:
            return self._known_state_space.pos_to_idx(positions[:, self._known_factor_indices])
        else:
            raise IndexError(f'positions has incorrect number of dimensions: {positions.ndim}')

    def idx_to_pos(self, indices):
        indices = np.array(indices)
        if indices.ndim == 0:
            known_factors = self._known_state_space.idx_to_pos(indices.reshape(-1))
            sampled_factors = self._full_state_space.sample_missing_factors(known_factors, self._known_factor_indices)[0]
        elif indices.ndim == 1:
            known_factors = self._known_state_space.idx_to_pos(indices)
            sampled_factors = self._full_state_space.sample_missing_factors(known_factors, self._known_factor_indices)
        else:
            raise IndexError(f'indices has incorrect number of dimensions: {indices.ndim}')
        return sampled_factors
        
    def sample_factors(self, num_samples, factor_indices=None):
        return self._full_state_space.sample_factors(
            num_samples,
            self._known_factor_indices[factor_indices] if factor_indices else None
        )
    
    def sample_missing_factors(self, partial_factors, partial_factor_indices):
        return self._full_state_space.sample_missing_factors(
            partial_factors,
            self._known_factor_indices[partial_factor_indices]
        )

    def resample_factors(self, factors, fixed_factor_indices):
        return self._full_state_space.resample_factors(
            factors,
            self._known_factor_indices[fixed_factor_indices]
        )


# ========================================================================= #
# Hidden State Space                                                        #
# ========================================================================= #


class StateSpaceRemapIndex(object):
    """Mapping from incorrectly ordered factors to state space indices"""
    
    def __init__(self, factor_sizes, features):
        self._states = StateSpace(factor_sizes)
        # get indices of features
        orig_indices = self._states.pos_to_idx(features)
        if np.unique(orig_indices).size != self._states.size:
            raise ValueError("Features do not cover the entire state space.")
        # get indices of state space
        state_indices = np.arange(self._states.size)
        # mapping
        self._state_to_orig_idx = np.zeros(self._states.size, dtype=np.int64)
        self._state_to_orig_idx[orig_indices] = state_indices
    
    def factors_to_orig_idx(self, factors):
        """
        get the original index of factors
        """
        return self._state_to_orig_idx[self._states.pos_to_idx(factors)]
