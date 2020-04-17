import numpy as np


# ========================================================================= #
# index                                                                   #
# ========================================================================= #


class SplitDiscreteStateSpace(object):
    """
    State space with factors split between latent variable and observations.
    ie. multi dimensional index which samples from unused dimensions.
    MODIFIED FROM: https://github.com/google-research/disentanglement_lib/blob/adb2772b599ea55c60d58fd4b47dff700ef9233b/disentanglement_lib/data/ground_truth/util.py
    """

    def __init__(self, factor_sizes, latent_factor_indices):
        # dimension sizes
        self._factor_sizes = np.array(factor_sizes)
        self._size = np.prod(self._factor_sizes)
        # indices of used dimensions
        self._latent_factor_indices = latent_factor_indices
        self._latent_size = np.prod(self._factor_sizes[self._latent_factor_indices])
        # indices of unused dimensions
        self._observation_factor_indices = [i for i in range(self.num_factors) if i not in self._latent_factor_indices]
        # helper data for conversion between factors and indexes
        bases = np.prod(self._factor_sizes) // np.cumprod([1, *self._factor_sizes])
        self._factor_divisors = bases[1:]
        self._factor_modulus = bases[:-1]

    @property
    def num_factors(self):
        return len(self._factor_sizes)

    @property
    def size(self):
        return self._size

    @property
    def latent_size(self):
        return self._size

    @property
    def num_latent_factors(self):
        return len(self._latent_factor_indices)

    def factor_to_idx(self, pos):
        return np.dot(pos, self._factor_divisors)

    def idx_to_factor(self, idx):
        return (idx % self._factor_modulus) // self._factor_divisors

    def sample_latent_factors(self, num_samples):
        """
        Sample a batch of the latent factors.
        ie. only returns factors from used dimensions.
        """
        # factors = np.zeros(shape=(num, self.num_latent_factors), dtype=np.int64)
        # for i, idx in enumerate(self._latent_factor_indices):
        #     factors[:, i] = self._sample_factor(idx, num)
        # return factors
        return self._sample_factors(num_samples, self._latent_factor_indices)

    def sample_all_factors(self, latent_factors):
        """
        Samples the remaining factors based on the latent factors.
        ie. fills in the missing factors around latent_factors by sampling from the unused dimensions.
        """
        num_samples = latent_factors.shape[0]
        all_factors = np.zeros(shape=(num_samples, self.num_factors), dtype=np.int64)
        all_factors[:, self._latent_factor_indices] = latent_factors
        # Complete all the other factors
        # for idx in self._observation_factor_indices:
        #     all_factors[:, idx] = self._sample_factor(idx, num_samples)
        all_factors[:, self._observation_factor_indices] = self._sample_factors(self._observation_factor_indices, num_samples)
        return all_factors

    def resample_all_factors(self, factors):
        """Resample across all the factors, keeping latent factors constant."""
        return self.sample_all_factors(factors[:, self._latent_factor_indices])

    # def _sample_factor(self, i, num):
    #     return np.random.randint(self._factor_sizes[i], size=num)

    def _sample_factors(self, factor_indices, num_samples):
        """return the specified number of position samples based on a list of dimension sizes."""
        return np.random.randint(
            self._factor_sizes[factor_indices],
            size=(num_samples, len(factor_indices))
        )

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
