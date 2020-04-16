import numpy as np


# ========================================================================= #
# index                                                                   #
# ========================================================================= #


class MultiDimIndex(object):
    def __init__(self, dimension_sizes, used_dimensions=None):
        # dimensions
        self._dimension_sizes = dimension_sizes
        self._size = np.prod(dimension_sizes)
        # used dimensions
        self._used_dimensions = used_dimensions if used_dimensions else list(range(self.num_dimensions))
        self._used_dimension_sizes = [self._dimension_sizes[i] for i in self._used_dimensions]
        # helper data for conversion between positions and indexes
        bases = np.prod(self._dimension_sizes) // np.cumprod([1, *self._dimension_sizes])
        self._pos_divisors = bases[1:]
        self._pos_modulus = bases[:-1]

    @property
    def num_dimensions(self):
        return len(self._dimension_sizes)

    @property
    def num_used_dimensions(self):
        return len(self._used_dimensions)

    @property
    def size(self):
        return self._size

    def pos_to_idx(self, pos):
        return np.dot(pos, self._pos_divisors)

    def idx_to_pos(self, idx):
        return (idx % self._pos_modulus) // self._pos_divisors

    def _sample_dimensions(self, num_samples, dimension_sizes):
        """return the specified number of position samples based on a list of dimension sizes."""
        return np.random.randint(
            dimension_sizes,
            size=(num_samples, len(dimension_sizes))
        )

    def pos_sample(self, num_samples, fixed_dimensions=None):
        """Sample across all the dimensions and return the positions."""
        fixed_dimension_sizes = self._used_dimension_sizes if (fixed_dimensions is None) else [self._dimension_sizes[i] for i in fixed_dimensions]
        return self._sample_dimensions(num_samples, fixed_dimension_sizes)

    def pos_sample_fixed(self, fixed_positions, fixed_dimensions=None):
        """Sample across the missing dimensions and return the positions."""
        num_samples = fixed_positions.shape[0]
        positions = np.zeros(shape=(num_samples, self.num_dimensions), dtype=np.int64)
        # set constant values for samples
        if fixed_dimensions is None:
            fixed_dimensions = self._used_dimensions
        positions[:, fixed_dimensions] = fixed_positions
        # sample the remaining factors that are not fixed
        sample_dimensions = [i for i in range(self.num_dimensions) if i not in fixed_dimensions]
        sample_dimension_sizes = [self._dimension_sizes[i] for i in sample_dimensions]
        positions[:, sample_dimensions] = self._sample_dimensions(num_samples, sample_dimension_sizes)
        return positions

    def pos_resample(self, positions, fixed_dimensions=None):
        """Resample across the non-fixed dimensions and return the positions."""
        if fixed_dimensions is None:
            fixed_dimensions = self._used_dimensions
        return self.pos_sample_fixed(positions[:, fixed_dimensions], fixed_dimensions)

    def idx_sample(self, num_samples, fixed_dimensions=None):
        """Sample across all the dimensions and return the indices."""
        return self.pos_to_idx(self.pos_sample(num_samples, fixed_dimensions))

    def idx_sample_fixed(self, fixed_positions, fixed_dimensions=None):
        """Sample across the missing dimensions and return the indices."""
        return self.pos_to_idx(self.pos_sample_fixed(fixed_positions, fixed_dimensions))

    def idx_resample(self, positions, fixed_dimensions=None):
        """Resample across the non-fixed dimensions and return the indices."""
        return self.pos_to_idx(self.pos_resample(positions, fixed_dimensions))


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

