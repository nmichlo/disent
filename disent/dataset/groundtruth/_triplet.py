import logging
import numpy as np

from disent.data.groundtruth.base import GroundTruthData
from disent.dataset.groundtruth import GroundTruthDataset

log = logging.getLogger(__name__)


# ========================================================================= #
# triplets                                                                  #
# ========================================================================= #


class GroundTruthDatasetTriples(GroundTruthDataset):

    def __init__(
            self,
            ground_truth_data: GroundTruthData,
            transform=None,
            # factor sampling
            p_k_range=(1, -2),
            n_k_range=(1, -1),
            n_k_is_offset=True,
            share_p=True,
            # radius sampling
            p_radius_range=(1, -2),
            n_radius_range=(2, -1),
            # final checks
            swap_metric=None,
    ):
        super().__init__(
            ground_truth_data=ground_truth_data,
            transform=transform,
        )
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # DIFFERING FACTORS
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.p_min, self.p_max = normalise_range_pair(p_k_range, self.data.num_factors)
        self.n_min, self.n_max = normalise_range_pair(n_k_range, self.data.num_factors)
        self.n_k_is_offset = n_k_is_offset
        self.share_p = share_p
        # cross factor assertions
        if not n_k_is_offset:
            assert self.p_min <= self.n_min
            assert self.p_max <= self.n_max
        else:
            assert (self.p_max + self.n_min) <= self.data.num_factors
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # RADIUS SAMPLING
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.p_radius_min, self.p_radius_max = normalise_range_pair(p_radius_range, self.data.factor_sizes)
        self.n_radius_min, self.n_radius_max = normalise_range_pair(n_radius_range, self.data.factor_sizes)
        assert np.all(self.p_min <= self.n_min)
        assert np.all(self.p_max <= self.n_max)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # OTHER
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # # if _n_radius should be offset from _p_radius, rather than index 0
        # if n_resample_is_offset:
        #     raise NotImplemented('n_resample_is_offset has not yet been implemented')
        # # values
        # self._p_radius = p_radius_range
        # self._n_radius = n_radius_range
        # swap if the sampled factors are ordered wrong
        self._swap_metric = swap_metric
        assert swap_metric in {None, 'factors', 'manhattan', 'manhattan_ratio'}

    def __getitem__(self, idx):
        f0, f1, f2 = self.sample_factors(idx)
        obs0 = self._getitem_transformed(self.data.pos_to_idx(f0))
        obs1 = self._getitem_transformed(self.data.pos_to_idx(f1))
        obs2 = self._getitem_transformed(self.data.pos_to_idx(f2))
        return obs0, obs1, obs2

    def sample_factors(self, idx):
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # NUM DIFFERING FACTORS
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        if self.n_k_is_offset:
            # Sample so that [n_min, n_max] is offset from p_num
            p_num_k = np.random.randint(self.p_min, self.p_max + 1)
            n_num_k = np.random.randint(p_num_k + self.n_min, min(p_num_k + self.n_max, self.data.num_factors) + 1)
        else:
            # Sample [p_min, p_max] and [n_min, n_max] individually
            p_num_k = np.random.randint(self.p_min, self.p_max + 1)
            n_num_k = np.random.randint(max(p_num_k, self.n_min), self.n_max + 1)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # SHARED FACTOR INDICES
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        p_num_shared = self.data.num_factors - p_num_k
        n_num_shared = self.data.num_factors - n_num_k
        # sample
        if self.share_p:
            p_shared_indices = np.random.choice(self.data.num_factors, size=p_num_shared, replace=False)
            n_shared_indices = p_shared_indices[:n_num_shared]
        else:
            p_shared_indices = np.random.choice(self.data.num_factors, size=p_num_shared, replace=False)
            n_shared_indices = np.random.choice(self.data.num_factors, size=n_num_shared, replace=False)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # RESAMPLE FACTORS
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        anchor_factors = self.data.idx_to_pos(idx)
        positive_factors = self.data.resample_radius(anchor_factors, resample_radius=self._p_radius, distinct=True, shared_factor_indices=p_shared_indices)
        negative_factors = self.data.resample_radius(anchor_factors, resample_radius=self._n_radius, distinct=True, shared_factor_indices=n_shared_indices)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # SWAP IF +VE FURTHER THAN -VE
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        if self._swap_metric is not None:
            positive_factors, negative_factors = self._swap_factors(anchor_factors, positive_factors, negative_factors)
        # return observations
        return anchor_factors, positive_factors, negative_factors

    def _swap_factors(self, anchor_factors, positive_factors, negative_factors):
        if self._swap_metric == 'factors':
            p_dist = np.sum(anchor_factors == positive_factors)
            n_dist = np.sum(anchor_factors == negative_factors)
        elif self._swap_metric == 'manhattan':
            p_dist = np.sum(np.abs(anchor_factors - positive_factors))
            n_dist = np.sum(np.abs(anchor_factors - negative_factors))
        elif self._swap_metric == 'manhattan_ratio':
            p_dist = np.sum(np.abs((anchor_factors - positive_factors) / self.data.factor_sizes))
            n_dist = np.sum(np.abs((anchor_factors - negative_factors) / self.data.factor_sizes))
        elif self._swap_metric == 'euclidean':
            p_dist = np.linalg.norm(anchor_factors - positive_factors)
            n_dist = np.linalg.norm(anchor_factors - negative_factors)
        elif self._swap_metric == 'euclidean_ratio':
            p_dist = np.linalg.norm((anchor_factors - positive_factors) / self.data.factor_sizes)
            n_dist = np.linalg.norm((anchor_factors - negative_factors) / self.data.factor_sizes)
        else:
            raise KeyError
        # perform swap
        if p_dist < n_dist:
            positive_factors, negative_factors = negative_factors, positive_factors
            log.warning(f'Swapped factors based on metric: {self._swap_metric}')
        # return factors
        return positive_factors, negative_factors


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


def normalise_range(mins, maxs, sizes):
    sizes = np.array(sizes)
    # compute the bounds for each factor
    mins, maxs = np.broadcast_to(mins, sizes.shape).copy(), np.broadcast_to(maxs, sizes.shape).copy()
    mins[mins < 0] += sizes[mins < 0] + 1
    maxs[maxs < 0] += sizes[maxs < 0] + 1
    # check that min <= max
    assert np.all(mins <= maxs)
    # check that everything is in the right range [1, -1]
    assert np.all(0 <= mins) and np.all(mins <= sizes)
    assert np.all(0 <= maxs) and np.all(maxs <= sizes)
    # return merged
    return mins, maxs


def normalise_range_pair(min_max, sizes):
    min_max = np.array(min_max)
    # if not a 2 tuple, repeat. This fixes the min == max.
    if min_max.shape == ():
        min_max = min_max.repeat(2)
    # check final shape
    assert min_max.shape == (2,)
    # get values
    return normalise_range(*min_max, sizes)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
