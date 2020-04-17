from typing import Optional, Tuple
from torch.utils.data.dataset import Dataset, IterableDataset
import numpy as np


# ========================================================================= #
# index                                                                     #
# ========================================================================= #


class MultiDimIndex(object):

    def __init__(self, dimension_sizes):
        # dimension
        self._dimension_sizes = dimension_sizes
        self._size = np.prod(dimension_sizes)
        # helper data for conversion between factors and indexes
        bases = np.prod(self._dimension_sizes) // np.cumprod([1, *self._dimension_sizes])
        self._pos_divisors = bases[1:]
        self._pos_modulus = bases[:-1]

    @property
    def size(self):
        return self._size

    def pos_to_idx(self, pos):
        return np.dot(pos, self._pos_divisors)

    def idx_to_pos(self, idx):
        return (idx % self._pos_modulus) // self._pos_divisors


# ========================================================================= #
# split state space                                                         #
# ========================================================================= #


class SplitDiscreteStateSpace(object):
    """
    State space with factors split between latent variable and observations.
    ie. multi dimensional index which samples from unused dimensions.
    MODIFIED FROM: https://github.com/google-research/disentanglement_lib/blob/adb2772b599ea55c60d58fd4b47dff700ef9233b/disentanglement_lib/data/ground_truth/util.py
    """

    def __init__(self, factor_sizes, latent_factor_indices=None):
        # dimension sizes
        self._factor_sizes = np.array(factor_sizes)
        # indices of used dimensions
        self._latent_factor_indices = latent_factor_indices if latent_factor_indices else list(range(self.num_factors))
        # indices of unused dimensions
        self._observation_factor_indices = [i for i in range(self.num_factors) if i not in self._latent_factor_indices]
        # conversion between factors and indexes
        self._factor_index = MultiDimIndex(self._factor_sizes)
        self._latent_factor_index = MultiDimIndex(self._factor_sizes[self._latent_factor_indices])

    @property
    def num_factors(self):
        return len(self._factor_sizes)

    @property
    def num_latent_factors(self):
        return len(self._latent_factor_indices)

    @property
    def size(self):
        return self._factor_index.size

    @property
    def latent_size(self):
        return self._latent_factor_index.size

    def factor_to_idx(self, factor): return self._factor_index.pos_to_idx(factor)
    def idx_to_factor(self, idx): return self._factor_index.idx_to_pos(idx)

    def latent_factor_to_latent_idx(self, factor): return self._latent_factor_index.pos_to_idx(factor)
    def latent_idx_to_latent_factor(self, idx): return self._latent_factor_index.idx_to_pos(idx)

    def sample_latent_factors(self, num_samples):
        """
        Sample a batch of the latent factors.
        ie. only returns factors from used dimensions.
        """
        # factors = np.zeros(shape=(num, self.num_latent_factors), dtype=np.int64)
        # for i, idx in enumerate(self._latent_factor_indices):
        #     factors[:, i] = self._sample_factor(idx, num)
        # return factors
        return self._sample_factors(self._latent_factor_indices, num_samples)

    def sample_all_factors(self, latent_factors):
        """
        Samples the remaining factors based on the latent factors.
        ie. fills in the missing factors around latent_factors by sampling from the unused dimensions.
        """
        # set used factors
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self._latent_factor_indices] = latent_factors
        # sample for unused factors
        all_factors[:, self._observation_factor_indices] = self._sample_factors(self._observation_factor_indices, num_samples)
        return all_factors

    def resample_all_factors(self, factors):
        """Resample across all the factors, keeping latent factors constant."""
        return self.sample_all_factors(factors[:, self._latent_factor_indices])

    def _sample_factors(self, factor_indices, num_samples):
        """return the specified number of position samples based on a list of dimension sizes."""
        return np.random.randint(
            self._factor_sizes[factor_indices],
            size=(num_samples, len(factor_indices))
        )


# ========================================================================= #
# ground truth data                                                         #
# ========================================================================= #


class GroundTruthData(object):

    def __init__(self):
        assert len(self.factor_names) == len(
            self.factor_sizes), 'Dimensionality mismatch of FACTOR_NAMES and FACTOR_DIMS'
        self._state_space = SplitDiscreteStateSpace(
            factor_sizes=self.factor_sizes,
            latent_factor_indices=self.used_factors,
        )

    @property
    def size(self):
        return self._state_space.size

    @property
    def latent_size(self):
        return self._state_space.latent_size

    @property
    def num_factors(self):
        return self._state_space.num_factors

    @property
    def num_latent_factors(self):
        return self._state_space.num_latent_factors

    @property
    def factor_names(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def used_factors(self) -> Optional[Tuple[int, ...]]:
        raise NotImplementedError()

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    def sample(self, num_samples):
        """Sample a batch of factors Y and observations X."""
        latent_factors = self.sample_latent_factors(num_samples)
        return latent_factors, self.sample_observations_from_factors(latent_factors)

    def sample_observations(self, num_samples):
        """Sample a batch of observations X."""
        factors, observations = self.sample(num_samples)
        return observations

    def sample_latent_factors(self, num_samples):
        """Sample a batch of factors Y."""
        latent_factors = self._state_space.sample_latent_factors(num_samples)
        return latent_factors

    def sample_indices_from_factors(self, latent_factors):
        all_factors = self._state_space.sample_all_factors(latent_factors)
        return self._state_space.factor_to_idx(all_factors)

    def sample_observations_from_factors(self, latent_factors):
        """Sample a batch of observations X given a batch of factors Y."""
        indices = self.sample_indices_from_factors(latent_factors)
        return self.get_observations_from_indices(indices)

    def get_observations_from_indices(self, indices):
        raise NotImplementedError()


# ========================================================================= #
# paired factor of variation dataset                                        #
# ========================================================================= #

# """
# TODO:
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

class PairedVariationDataset(IterableDataset):

    def __init__(self, dataset, k):
        assert isinstance(dataset, GroundTruthData) and isinstance(dataset, Dataset), 'passed object is not an instance of both GroundTruthData and Dataset'
        self._dataset = dataset
        self._k = k

    def __iter__(self):
        for i in range(len(self._dataset)):
            # get fixed or random k
            k = np.random.randint(1, self._dataset.num_factors) if self._k == 'uniform' else self._k
            # make k random indices not shared
            shared_indices = np.random.choice(self._dataset.num_latent_factors, size=self._dataset.num_latent_factors - k, replace=False)
            # get shared factors d - k
            shared_factors = self._dataset.sample_latent_factors(1)[0]
            # TODO: make use of this:
            # shared_factors = shared_factors[0, shared_indices]
            # resample not shared factors
            # TODO: this could generate the same factors due to randomness
            # TODO: this isnt actually sampling the right thing
            idx1, idx2 = self._dataset.sample_indices_from_factors(
                np.array([shared_factors, shared_factors])
            )
            x1, x2 = self._dataset.__getitem__(idx1), self._dataset.__getitem__(idx2)
            yield x1, x2





# ========================================================================= #
# END                                                                       #
# ========================================================================= #
