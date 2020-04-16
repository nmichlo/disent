from torch.utils.data import Dataset
import abc
import numpy as np


# ========================================================================= #
# variation_factors                                                                   #
# ========================================================================= #


class SplitDiscreteStateSpace(object):
    """State space with factors split between latent variable and observations."""

    def __init__(self, factor_dimension_sizes, latent_factor_indices):
        """

        :param factor_sizes:
        :param latent_factor_indices:
        """
        self._factor_dim_sizes = np.array(factor_dimension_sizes)
        self._latent_factor_indices = latent_factor_indices
        self._latent_factor_dim_sizes = self._factor_dim_sizes[self._latent_factor_indices]
        self._observation_factor_indices = [i for i in range(self.num_factors) if i not in self._latent_factor_indices]
        self._observation_factor_dim_sizes = self._factor_dim_sizes[self._observation_factor_indices]

    @property
    def num_factors(self):
        return len(self._factor_dim_sizes)

    @property
    def num_latent_factors(self):
        return len(self._latent_factor_indices)

    def sample_latent_factors(self, num_samples):
        """
        Sample a batch of the latent factors.
        returns: integer array of shape (num_samples, num_latent_factors)
        """
        return np.random.randint(
            self._latent_factor_dim_sizes,
            size=(num_samples, self.num_latent_factors)
        )

    def sample_remaining_factors(self, latent_factors):
        """Samples the remaining factors based on the values in the latent factors."""

        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(shape=(num_samples, self.num_factors), dtype=np.int64)

        # set constant values for samples
        all_factors[:, self._latent_factor_indices] = latent_factors

        # sample the remaining factors not included in the latent space
        all_factors[:, self._observation_factor_indices] = np.random.randint(
            self._observation_factor_dim_sizes,
            size=(num_samples, len(self._observation_factor_indices))
        )

        return all_factors





class MultiDimIndex(object):

    def __init__(self, dimension_sizes, dimension_names=None):
        self._dimension_sizes = dimension_sizes
        self._dimension_names = dimension_names if dimension_names else [f'dim{i}' for i in range(self.num_dimensions)]
        self._size = np.prod(dimension_sizes)
        # helper data for conversion between positions and indexes
        bases = np.prod(self._dimension_sizes) // np.cumprod([1, *self._dimension_sizes])
        self._pos_divisors = bases[1:]
        self._pos_modulus = bases[:-1]

    def sample_observations_from_factors(self, factors):
        all_factors = self.state_space.sample_all_factors(factors)
        indices = np.array(np.dot(all_factors, self.factor_bases), dtype=np.int64)
        return self.images[indices]

    @property
    def num_dimensions(self):
        return len(self._dimension_sizes)

    def pos_to_idx(self, pos):
        return np.dot(pos, self._pos_divisors)

    def idx_to_pos(self, idx):
        return (idx % self._pos_modulus) // self._pos_divisors

    def __len__(self):
        return self._size

    def _sample_dimensions(self, num_samples, dimension_sizes):
        """return the specified number of position samples based on a list of dimension sizes"""
        return np.random.randint(
            dimension_sizes,
            size=(num_samples, len(dimension_sizes))
        )

    def sample_pos(self, num_samples):
        """Sample across all the dimensions and return the positions"""
        return self._sample_dimensions(num_samples, self._dimension_sizes)

    def sample_idx(self, num_samples):
        """Sample across all the dimensions and return the indexes"""
        return self.pos_to_idx(self.sample_pos(num_samples))

    def fixed_sample_pos(self, ):





# class GroundTruthDataset(object):
#     """Abstract class for data sets that are two-step generative models."""
#
#     @property
#     def num_factors(self):
#         raise NotImplementedError()
#
#     @property
#     def factor_dims(self):
#         raise NotImplementedError()
#
#     @property
#     def observation_shape(self):
#         raise NotImplementedError()
#
#     def sample_factors(self, num):
#         """Sample a batch of factors Y."""
#         raise NotImplementedError()
#
#     def sample_observations_from_factors(self, factors):
#         """Sample a batch of observations X given a batch of factors Y."""
#         raise NotImplementedError()
#
#     def sample(self, num):
#         """Sample a batch of factors Y and observations X."""
#         factors = self.sample_factors(num, random_state)
#         return factors, self.sample_observations_from_factors(factors, random_state)
#
#     def sample_observations(self, num):
#         """Sample a batch of observations X."""
#         return self.sample(num, random_state)[1]


# """
# Excerpt from Weakly-Supervised Disentanglement Without Compromises:
# [section 5. Experimental results]
#
# CREATE DATA SETS: with weak supervision from the existing
# disentanglement data sets:
# 1. we first sample from the discrete z according to the ground-truth generative model (1)–(2).
# 2. Then, we sample k factors of variation that should not be shared by the two images and re-sample those coordinates to obtain z˜.
#    This ensures that each image pair differs in at most k factors of variation.
#
# For k we consider the range from 1 to d − 1.
# This last setting corresponds to the case where all but one factor of variation are re-sampled.
#
# We study both the case where k is constant across all pairs in the data set and where k is sampled uniformly in the range [d − 1] for every training pair (k = Rnd in the following).
# Unless specified otherwise, we aggregate the results for all values of k.
# """

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
