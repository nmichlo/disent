from typing import Tuple
from torch.utils.data.dataset import Dataset
import numpy as np


# ========================================================================= #
# index                                                                     #
# ========================================================================= #


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
        self._size = np.prod(factor_sizes)
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

    @property
    def size(self):
        return self._size

    @property
    def num_factors(self):
        return len(self._factor_sizes)

    def pos_to_idx(self, pos):
        return np.dot(pos, self._factor_divisors)

    def idx_to_pos(self, idx):
        return (idx % self._factor_modulus) // self._factor_divisors

    def sample_factors(self, num_samples, factor_indices=None):
        """
        sample randomly from all factors, otherwise the given factor_indices.
        returned values appear in the same order as factor_indices.
        """
        return np.random.randint(
            self._factor_sizes if (factor_indices is None) else self._factor_sizes[factor_indices],
            size=(num_samples, len(factor_indices))
        )

    # def sample_indices(self, num_samples):
    #     """Like sample_factors but returns indices."""
    #     return self.pos_to_idx(self.sample_factors(num_samples))

    def sample_missing_factors(self, values, factor_indices):
        """
        Samples the remaining factors not given in the dimension_indices.
        ie. fills in the missing values by sampling from the unused dimensions.
        returned values are ordered by increasing factor index and not factor_indices.
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


# ========================================================================= #
# ground truth data                                                         #
# ========================================================================= #


class GroundTruthDataset(DiscreteStateSpace, Dataset):

    def __init__(self):
        assert len(self.factor_names) == len(self.factor_sizes), 'Dimensionality mismatch of FACTOR_NAMES and FACTOR_DIMS'
        super().__init__(self.factor_sizes)

    def sample_observations(self, num_samples):
        """Sample a batch of observations X."""
        factors = self.sample_factors(num_samples)
        indices = self.pos_to_idx(factors)
        return self[indices]

    @property
    def factor_names(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    def __getitem__(self, indices):
        """
        should return a single observation if an integer index, or
        an array of observations if indices is an array.
        """
        raise NotImplementedError()


# ========================================================================= #
# paired factor of variation dataset                                        #
# ========================================================================= #


class PairedVariationDataset(Dataset):

    def __init__(self, dataset: GroundTruthDataset, k=None, variation_factor_indices=None):
        """
        Dataset that pairs together samples with at most k differing factors of variation.

        dataset: A dataset that extends GroundTruthDataset
        k: An integer (k), None (k=d-1), or "uniform" (random k in range 1 to d-1)
        variation_factor_indices: The indices of the factors of variation that are samples between pairs, if None (all factors are sampled)
        """

        assert isinstance(dataset, GroundTruthDataset), 'passed object is not an instance of both GroundTruthData and Dataset'
        # wrapped dataset
        self._dataset: GroundTruthDataset = dataset
        # possible fixed dimensions between pairs
        self._variation_factor_indices = np.arange(self._dataset.num_factors) if (variation_factor_indices is None) else np.array(variation_factor_indices)
        # d
        self._num_variation_factors = len(self._variation_factor_indices)
        # number of varied factors between pairs
        self._k = self._num_variation_factors - 1 if (k is None) else k
        # verify k
        assert isinstance(k, str) or isinstance(k, int), f'k must be "uniform" or an integer 1 <= k <= d-1, d={self._num_variation_factors}'
        if isinstance(k, int):
            assert 1 <= k, 'k cannot be less than 1'
            assert k < self._num_variation_factors, f'all factors cannot be varied for each pair, k must be less than {self._num_variation_factors}'

    def __len__(self):
        # TODO: is dataset as big as the latent space OR as big as the orig.
        # return self._latent_space.size
        return self._dataset.size

    def __getitem__(self, idx):
        orig_factors, paired_factors = self.sample_pair_factors(idx)
        return self._dataset[[orig_factors, paired_factors]]

    def sample_pair_factors(self, idx):
        """
        Excerpt from Weakly-Supervised Disentanglement Without Compromises:
        [section 5. Experimental results]

        CREATE DATA SETS: with weak supervision from the existing
        disentanglement data sets:
        1. we first sample from the discrete z according to the ground-truth generative model (1)–(2).
        2. Then, we sample k factors of variation that should not be shared by the two images and re-sample those coordinates to obtain z˜.
           This ensures that each image pair differs in at most k factors of variation.

        For k we consider the range from 1 to d − 1.
        This last setting corresponds to the case where all but one factor of variation are re-sampled.

        We study both the case where k is constant across all pairs in the data set and where k is sampled uniformly in the range [d − 1] for every training pair (k = Rnd in the following).
        Unless specified otherwise, we aggregate the results for all values of k.
        """

        # get factors corresponding to index
        orig_factors = self._dataset.idx_to_pos(idx)
        # get fixed or random k
        k = np.random.randint(1, self._dataset.num_factors) if self._k == 'uniform' else self._k
        # make k random indices not shared
        num_shared = self._num_variation_factors - k
        shared_indices = np.random.choice(self._variation_factor_indices, size=num_shared, replace=False)
        # resample paired item, differs by at most k factors of variation
        paired_factors = self._dataset.resampled_factors(orig_factors[np.newaxis, :], shared_indices)
        # return observations
        return orig_factors, paired_factors[0]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
