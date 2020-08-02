import numpy as np


class DiscreteStateSpace(object):
    """
    State space where an index corresponds to coordinates in the factor space.
    ie. State space with multiple factors of variation, where each factor can be a different size.
    Heavily modified FROM: https://github.com/google-research/disentanglement_lib/blob/adb2772b599ea55c60d58fd4b47dff700ef9233b/disentanglement_lib/data/ground_truth/util.py
    """

    def __init__(self, factor_sizes):
        super().__init__()
        # dimension
        self._factor_sizes = np.array(factor_sizes)
        self._factor_sizes.flags.writeable = False
        self._size = int(np.prod(factor_sizes))
        # dimension sampling
        self._factor_indices_set = set(range(self.num_factors))
        # helper data for conversion between factors and indexes
        bases = np.prod(self._factor_sizes) // np.cumprod([1, *self._factor_sizes])
        self._factor_divisors = bases[1:]
        self._factor_modulus = bases[:-1]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.idx_to_pos(idx)

    def __iter__(self):
        for i in range(self.size):
            yield self[i]

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
        """
        Convert a position to an index (or convert a list of positions to a list of indices)
        - positions are lists of integers, with each element < their corresponding factor size
        - indices are integers < size
        """
        positions = np.array(positions).T
        return np.ravel_multi_index(positions, self._factor_sizes)

    def idx_to_pos(self, indices):
        """
        Convert an index to a position (or convert a list of indices to a list of positions)
        - indices are integers < size
        - positions are lists of integers, with each element < their corresponding factor size
        """
        positions = np.unravel_index(indices, self._factor_sizes)
        return np.array(positions).T

    def sample_factors(self, num_samples, factor_indices=None):
        """
        sample randomly from all factors, otherwise the given factor_indices.
        returned values appear in the same order as factor_indices.
        """
        sizes = self._factor_sizes if (factor_indices is None) else self._factor_sizes[factor_indices]
        return np.random.randint(sizes, size=(num_samples, len(sizes)))

    # def sample_indices(self, num_samples):
    #     """Like sample_factors but returns indices."""
    #     return self.pos_to_idx(self.sample_factors(num_samples))

    def sample_missing_factors(self, values, factor_indices):
        """
        Samples the remaining factors not given in the dimension_indices.
        ie. fills in the missing values by sampling from the unused dimensions.
        returned values are ordered by increasing factor index and not factor_indices.

        values correspond to factor indices.
        """
        num_samples, num_dims = values.shape
        used_indices_set = set(factor_indices)
        # assertions
        assert num_dims == len(factor_indices), 'dimension count mismatch'
        assert len(used_indices_set) == len(factor_indices), 'dimension indices are duplicated'
        # set used dimensions
        all_values = np.zeros(shape=(num_samples, self.num_factors), dtype=np.int64)
        all_values[:, factor_indices] = values
        # sample for missing
        missing_indices = list(self._factor_indices_set - used_indices_set)
        all_values[:, missing_indices] = self.sample_factors(num_samples=num_samples, factor_indices=missing_indices)
        # return
        return all_values

    # def sample_missing_indices(self, values, factor_indices):
    #     """Like sample_missing_factors but returns indices."""
    #     return self.pos_to_idx(self.sample_missing_factors(values, factor_indices))

    def resampled_factors(self, values, factors_indices):
        """
        Resample across all the factors, keeping factor_indices constant.
        returned values are ordered by increasing factor index and not factor_indices.
        """
        return self.sample_missing_factors(values[:, factors_indices], factors_indices)

    # def resampled_indices(self, values, factors_indices):
    #     """Like resampled_factors but returns indices."""
    #     return self.pos_to_idx(self.resampled_factors(values, factors_indices))