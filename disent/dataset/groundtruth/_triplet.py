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
            n_k_is_shared=True,
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
        self.p_k_min, self.p_k_max = normalise_range_pair(p_k_range, self.data.num_factors)
        self.n_k_min, self.n_k_max = normalise_range_pair(n_k_range, self.data.num_factors)
        self.n_k_is_offset = n_k_is_offset
        self.n_k_is_shared = n_k_is_shared
        # cross factor assertions
        if not n_k_is_offset:
            assert self.p_k_min <= self.n_k_min
            assert self.p_k_max <= self.n_k_max
        else:
            assert (self.p_k_max + self.n_k_min) <= self.data.num_factors
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # RADIUS SAMPLING
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.p_radius_min, self.p_radius_max = normalise_range_pair(p_radius_range, self.data.factor_sizes)
        self.n_radius_min, self.n_radius_max = normalise_range_pair(n_radius_range, self.data.factor_sizes)
        assert np.all(self.p_radius_min <= self.p_radius_max)
        assert np.all(self.n_radius_min <= self.n_radius_max)
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
            p_k = np.random.randint(self.p_k_min, self.p_k_max + 1)
            n_k = np.random.randint(p_k + self.n_k_min, min(p_k + self.n_k_max, self.data.num_factors) + 1)
        else:
            # Sample [p_min, p_max] and [n_min, n_max] individually
            p_k = np.random.randint(self.p_k_min, self.p_k_max + 1)
            n_k = np.random.randint(max(p_k, self.n_k_min), self.n_k_max + 1)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # SHARED FACTOR INDICES
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        p_num_shared = self.data.num_factors - p_k
        n_num_shared = self.data.num_factors - n_k
        # sample
        if self.n_k_is_shared:
            p_shared_indices = np.random.choice(self.data.num_factors, size=p_num_shared, replace=False)
            n_shared_indices = p_shared_indices[:n_num_shared]
        else:
            p_shared_indices = np.random.choice(self.data.num_factors, size=p_num_shared, replace=False)
            n_shared_indices = np.random.choice(self.data.num_factors, size=n_num_shared, replace=False)
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # RESAMPLE FACTORS
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        anchor_factors = self.data.idx_to_pos(idx)
        positive_factors = sample_radius(anchor_factors, 0, self.data.factor_sizes, self.p_radius_min, self.p_radius_max)
        negative_factors = sample_radius(anchor_factors, 0, self.data.factor_sizes, self.n_radius_min, self.n_radius_max)
        positive_factors[p_shared_indices] = anchor_factors[p_shared_indices]
        negative_factors[n_shared_indices] = negative_factors[n_shared_indices]
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


def randint2(a_low, a_high, b_low, b_high, size=None):
    """
    Like np.random.randint, but supports two ranges of values.
    Samples with equal probability from both ranges.
    - a: [a_low, a_high) -> including a_low, excluding a_high!
    - b: [b_low, b_high) -> including b_low, excluding b_high!
    """
    # convert
    a_low, a_high = np.array(a_low), np.array(a_high)
    b_low, b_high = np.array(b_low), np.array(b_high)
    # checks
    assert np.all(a_low <= a_high)
    assert np.all(b_low <= b_high)
    assert np.all(a_high <= b_low)
    # compute
    da = a_high - a_low
    db = b_high - b_low
    d = da + db
    assert np.all(d > 0), 'r'
    # sampled
    offset = np.random.randint(0, d, size=size)
    offset += (da <= offset) * (b_low - a_high)
    return a_low + offset


def sample_radius(factors, factor_mins, factor_maxs, radius_mins, radius_maxs):
    factors = np.array(factors)
    return randint2(
        a_low=np.maximum(factors - radius_maxs, factor_mins),
        a_high=factors - radius_mins + 1,
        # if radius_mins == 0, then the ranges overlap, so we must shift one of them.
        b_low=factors + radius_mins + (radius_mins == 0),
        b_high=np.minimum(factors + radius_maxs + 1, factor_maxs),
    )

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
