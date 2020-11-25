import numpy as np
from disent.dataset.groundtruth._single import GroundTruthDataset
from disent.data.groundtruth.base import GroundTruthData
from disent.dataset.groundtruth._triplet import sample_radius, normalise_range_pair, FactorSizeError


# ========================================================================= #
# paired ground truth dataset                                               #
# ========================================================================= #


class GroundTruthDatasetPairs(GroundTruthDataset):

    def __init__(
            self,
            ground_truth_data: GroundTruthData,
            transform=None,
            augment=None,
            # factor sampling
            p_k_range=(1, -1),
            # radius sampling
            p_radius_range=(1, -1),
    ):
        """
        Dataset that pairs together samples with at most k differing factors of variation.

        dataset: A dataset that extends GroundTruthData
        k: An integer (k), None (k=d-1), or "uniform" (random k in range 1 to d-1)
        variation_factor_indices: The indices of the factors of variation that are sampled between pairs, if None (all factors are sampled)
        """
        super().__init__(ground_truth_data=ground_truth_data, transform=transform, augment=augment)
        # DIFFERING FACTORS
        self.p_k_min, self.p_k_max = self._min_max_from_range(p_range=p_k_range, max_values=self.data.num_factors)
        # RADIUS SAMPLING
        self.p_radius_min, self.p_radius_max = self._min_max_from_range(p_range=p_radius_range, max_values=self.data.factor_sizes)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # CORE                                                                  #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def __getitem__(self, idx):
        f0, f1 = self.datapoint_sample_factors_pair(idx)
        return self.dataset_get_observation(
            self.data.pos_to_idx(f0),
            self.data.pos_to_idx(f1),
        )

    def datapoint_sample_factors_pair(self, idx):
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
        # SAMPLE FACTOR INDICES
        p_k = self._sample_num_factors()
        p_shared_indices = self._sample_shared_indices(p_k)
        # SAMPLE FACTORS - sample, resample and replace shared factors with originals
        anchor_factors = self.data.idx_to_pos(idx)
        positive_factors = self._resample_factors(anchor_factors)
        positive_factors[p_shared_indices] = anchor_factors[p_shared_indices]
        return anchor_factors, positive_factors

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _min_max_from_range(self, p_range, max_values):
        p_min, p_max = normalise_range_pair(p_range, max_values)
        # cross factor assertions
        if not np.all(p_max <= max_values):
            raise FactorSizeError('Factor dimensions are too small for given range:'
                                  f'\n\tUnsatisfied: p_max <= max_size'
                                  f'\n\tUnsatisfied: {p_max} <= {np.array(max_values)}')
        return p_min, p_max

    def _sample_num_factors(self):
        p_k = np.random.randint(self.p_k_min, self.p_k_max + 1)
        return p_k

    def _sample_shared_indices(self, p_k):
        p_shared_indices = np.random.choice(self.data.num_factors, size=self.data.num_factors-p_k, replace=False)
        return p_shared_indices

    def _resample_factors(self, anchor_factors):
        positive_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_low=self.p_radius_min, r_high=self.p_radius_max + 1)
        return positive_factors


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


# if __name__ == '__main__':
#     from disent.data.groundtruth import XYMultiGridData
#     from disent.util import concat_lines
#
#     # check that resample radius is working correctly!
#     dataset = GroundTruthDatasetPairs(
#         XYMultiGridData(1, 4),
#         p_k_range=(1, 1),
#         p_radius_range=(1, 1)
#     )
#
#     for pair in dataset:
#         obs0, obs1 = np.array(pair[0], dtype='int'), np.array(pair[1], dtype='int')
#         # CHECKS
#         diff = np.abs(obs1 - obs0)
#         diff_coords = np.array(np.where(diff > 0)).T
#         assert len(diff_coords) == 2  # check max changes
#         dist = np.abs(diff_coords[0] - diff_coords[1])
#         assert np.sum(dist > 0) == 1  # check max changes
#         assert np.max(dist) == 1      # check radius
#         # INFO
#         print(concat_lines(*[((obs > 0) * [1, 2, 4]).sum(axis=-1) for obs in (obs0, obs1)]), '\n')
