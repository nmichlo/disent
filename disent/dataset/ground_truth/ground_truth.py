from disent.dataset.data_index import UsedMultiDimIndex
from typing import Optional, Tuple
from torch.utils.data import Dataset
import numpy as np


# ========================================================================= #
# ground truth dataset                                                      #
# ========================================================================= #


class GroundTruthDataset(Dataset):

    def __init__(self):
        assert len(self.FACTOR_NAMES) == len(
            self.FACTOR_DIMS), 'Dimensionality mismatch of FACTOR_NAMES and FACTOR_DIMS'
        self._index = UsedMultiDimIndex(
            dimension_sizes=self.FACTOR_DIMS,
            used_dimensions=self.USED_DIMS,
        )

    @property
    def num_factors(self):
        return self._index.num_dimensions

    @property
    def __len__(self):
        return self._index.size

    @property
    def FACTOR_NAMES(self) -> Tuple[str, ...]:
        raise NotImplementedError()
    @property
    def FACTOR_DIMS(self) -> Tuple[int, ...]:
        raise NotImplementedError()
    @property
    def USED_DIMS(self) -> Optional[Tuple[int, ...]]:
        raise NotImplementedError()
    @property
    def OBSERVATION_SHAPE(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    def sample(self, num_samples):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num_samples)
        return factors, self.sample_observations_from_factors(factors)

    def sample_factors(self, num_samples):
        """return a number of sampled factors"""
        return self._index.sample(num_samples)

    def sample_observations(self, num_samples):
        """Sample a batch of observations X."""
        factors, observations = self.sample(num_samples)
        return observations

    def sample_observations_from_factors(self, factors):
        return self.get_observations_from_factors(self._index)

    def get_observations_from_factors(self, factors):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()


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
