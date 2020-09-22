from typing import Union, Optional
import numpy as np
from torch.utils.data import Dataset
from disent.dataset.groundtruth import GroundTruthDataset
from disent.util import LengthIter


# ========================================================================= #
# paired ground truth dataset                                               #
# ========================================================================= #


class GroundTruthDatasetPairs(Dataset, LengthIter):

    def __init__(
            self,
            dataset: GroundTruthDataset,
            k: int = 1,
            resample_radius: Optional[Union[str, int]] = 'inf',
            force_different_factors: bool = True,
            variation_factor_indices=None,
            return_factors: bool = False,
    ):
        """
        Dataset that pairs together samples with at most k differing factors of variation.

        dataset: A dataset that extends GroundTruthData
        k: An integer (k), None (k=d-1), or "uniform" (random k in range 1 to d-1)
        variation_factor_indices: The indices of the factors of variation that are sampled between pairs, if None (all factors are sampled)
        """
        assert isinstance(dataset, GroundTruthDataset), 'passed object is not an instance of GroundTruthDataset'
        assert len(dataset) > 1, 'Dataset must be contain more than one observation.'
        # wrapped dataset
        self._dataset = dataset
        self._data = dataset.data
        # possible fixed dimensions between pairs
        self._variation_factor_indices = np.arange(self._data.num_factors) if (variation_factor_indices is None) else np.array(variation_factor_indices)
        self._variation_factor_sizes = np.array(self._data.factor_sizes)[self._variation_factor_indices]
        self._num_variation_factors = len(self._variation_factor_indices)
        # number of varied factors between pairs
        self._k = self._num_variation_factors - 1 if (k is None) else k
        # verify k
        assert isinstance(self._k, str) or isinstance(self._k, int), f'k must be "uniform" or an integer 1 <= k <= d-1, d={self._num_variation_factors}'
        if isinstance(self._k, int):
            assert 1 <= self._k, 'k cannot be less than 1'
            assert self._k < self._num_variation_factors, f'all factors cannot be varied for each pair, k must be less than {self._num_variation_factors}'
        # if we must return (x, y) instead of just x, where y is the factors for x.
        self._return_factors = return_factors
        # if sampled factors MUST be different
        self._force_different_factors = force_different_factors
        # if we must sample according to offsets, rather than along an entire axis
        if resample_radius in {'inf', 'infinite', np.inf}:
            resample_radius = None
        assert (resample_radius is None) or (isinstance(resample_radius, int) and (resample_radius > 0))
        self._resample_radius = resample_radius

    def __len__(self):
        # TODO: is dataset as big as the latent space OR as big as the orig.
        # return self._latent_space.size
        return len(self._data)

    def __getitem__(self, idx):
        # get random factor pairs & get observations from factors
        factors = self.sample_factors(idx)
        observations = [self._dataset[self._data.pos_to_idx(pos)] for pos in factors]
        assert 2 <= len(observations) <= 3, 'More factors are not yet supported!'
        # return observations and factors, or just observations
        return list(zip(observations, factors)) if self._return_factors else observations

    def sample_factors(self, idx):
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
        orig_factors = self._data.idx_to_pos(idx)
        # get fixed or random k (k is number of factors that differ)
        k = np.random.randint(1, self._num_variation_factors) if (self._k == 'uniform') else self._k
        # return observations
        return orig_factors, self._resample_factors(orig_factors, k, self._resample_radius)

    def _resample_factors(self, base_factors, k, resample_radius):
        resampled_factors = None
        while (resampled_factors is None) or np.all(base_factors == resampled_factors):
            # make k random indices not shared + resample paired item, differs by at most k factors of variation
            num_shared = self._data.num_factors - k
            shared_indices = np.random.choice(self._variation_factor_indices, size=num_shared, replace=False)
            # how the non-shared indices are to be sampled
            if resample_radius is None:
                resampled_factors = self._data.resample_factors(base_factors[np.newaxis, :], shared_indices)[0]
            else:
                # elementwise sampling range for factors
                factors_min = np.maximum(base_factors - resample_radius, 0)
                factors_max = np.minimum(base_factors + resample_radius, self._variation_factor_sizes - 1)
                # choose factors & keep shared indices the same | TODO: this is inefficient sampling along all factors and then only keeping some
                resampled_factors = np.random.randint(factors_min, factors_max + 1)
                resampled_factors[shared_indices] = base_factors[shared_indices]
            # dont retry if sampled factors are the same
            if not self._force_different_factors:
                break
        return resampled_factors


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    from disent.data.groundtruth import XYMultiGridData
    from disent.util import concat_lines

    # check that resample radius is working correctly!
    dataset = XYMultiGridData(1, 4)
    dataset = GroundTruthDataset(dataset)
    dataset = GroundTruthDatasetPairs(dataset, resample_radius=1)
    for pair in dataset:
        obs0, obs1 = np.array(pair[0], dtype='int'), np.array(pair[1], dtype='int')
        # CHECKS
        diff = np.abs(obs1 - obs0)
        diff_coords = np.array(np.where(diff > 0)).T
        assert len(diff_coords) == 2  # check max changes
        dist = np.abs(diff_coords[0] - diff_coords[1])
        assert np.sum(dist > 0) == 1  # check max changes
        assert np.max(dist) == 1      # check radius
        # INFO
        print(concat_lines(*[((obs > 0) * [1, 2, 4]).sum(axis=-1) for obs in (obs0, obs1)]), '\n')
