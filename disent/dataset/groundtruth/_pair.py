from typing import Union, Optional
import numpy as np
from disent.dataset.groundtruth._single import GroundTruthDataset
from disent.data.groundtruth.base import GroundTruthData


# ========================================================================= #
# paired ground truth dataset                                               #
# ========================================================================= #


class GroundTruthDatasetPairs(GroundTruthDataset):

    def __init__(
            self,
            ground_truth_data: GroundTruthData,
            transform=None,
            # how the paired items are sampled
            k: int = 1,
            resample_radius: Optional[Union[str, int]] = 'inf',
    ):
        """
        Dataset that pairs together samples with at most k differing factors of variation.

        dataset: A dataset that extends GroundTruthData
        k: An integer (k), None (k=d-1), or "uniform" (random k in range 1 to d-1)
        variation_factor_indices: The indices of the factors of variation that are sampled between pairs, if None (all factors are sampled)
        """
        super().__init__(ground_truth_data, transform=transform)

        # number of varied factors between pairs
        self._k = self.data.num_factors - 1 if (k is None) else k
        assert isinstance(self._k, int) or (self._k == 'uniform'), f'{k=} must be "uniform" or an integer 1 <= k <= d-1, d={self.data.num_factors}'
        if isinstance(self._k, int):
            assert 1 <= self._k, 'k cannot be less than 1'
            assert self._k < self.data.num_factors, f'all factors cannot be varied for each pair, {k=} must be less than {self.data.num_factors}'

        # if we must sample according to offsets, rather than along an entire axis
        if resample_radius in {'inf', 'infinite', np.inf}:
            resample_radius = None
        assert (resample_radius is None) or (isinstance(resample_radius, int) and (resample_radius > 0)), f'{resample_radius=}'
        self._resample_radius = resample_radius

    def __getitem__(self, idx):
        # get random factor pair, then get observations from those factors
        return [
            self._getitem_transformed(self.data.pos_to_idx(pos))
            for pos in self.sample_factors(idx)
        ]

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
        orig_factors = self.data.idx_to_pos(idx)
        # get fixed or random k (k is number of factors that differ)
        k = np.random.randint(1, self.data.num_factors) if (self._k == 'uniform') else self._k
        # sample for new factors
        resampled_factors = self.data.resample_radius( orig_factors, resample_radius=self._resample_radius, distinct=True, num_shared_factors=self.data.num_factors-k)
        # return observations
        return orig_factors, resampled_factors


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    from disent.data.groundtruth import XYMultiGridData
    from disent.util import concat_lines

    # check that resample radius is working correctly!
    dataset = XYMultiGridData(1, 4)
    dataset = GroundTruthDatasetPairs(dataset, resample_radius=None)


    # for pair in dataset:
    #     obs0, obs1 = np.array(pair[0], dtype='int'), np.array(pair[1], dtype='int')
    #     # CHECKS
    #     diff = np.abs(obs1 - obs0)
    #     diff_coords = np.array(np.where(diff > 0)).T
    #     assert len(diff_coords) == 2  # check max changes
    #     dist = np.abs(diff_coords[0] - diff_coords[1])
    #     assert np.sum(dist > 0) == 1  # check max changes
    #     assert np.max(dist) == 1      # check radius
    #     # INFO
    #     print(concat_lines(*[((obs > 0) * [1, 2, 4]).sum(axis=-1) for obs in (obs0, obs1)]), '\n')