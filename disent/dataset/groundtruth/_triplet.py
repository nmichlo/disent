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
            n_k_sample_mode='offset',
            n_k_is_shared=True,
            # radius sampling
            p_radius_range=(1, -2),
            n_radius_range=(1, -1),
            n_radius_sample_mode='offset',
            # final checks
            swap_metric=None,
    ):
        super().__init__(ground_truth_data=ground_truth_data, transform=transform)
        # DIFFERING FACTORS
        self.n_k_sample_mode = n_k_sample_mode
        self.n_k_is_shared = n_k_is_shared
        self.p_k_min, self.p_k_max, self.n_k_min, self.n_k_max = self._min_max_from_range(p_range=p_k_range, n_range=n_k_range, max_values=self.data.num_factors, n_is_offset=self.n_k_is_offset)
        # RADIUS SAMPLING
        self.n_radius_sample_mode = n_radius_sample_mode
        self.p_radius_min, self.p_radius_max, self.n_radius_min, self.n_radius_max = self._min_max_from_range(p_range=p_radius_range, n_range=n_radius_range, max_values=self.data.factor_sizes, n_is_offset=self.n_radius_is_offset)
        # SWAP: if negative is not further than the positive
        self._swap_metric = swap_metric
        assert swap_metric in {None, 'factors', 'manhattan', 'manhattan_ratio', 'euclidean', 'euclidean_ratio'}

    def __getitem__(self, idx):
        f0, f1, f2 = self.sample_factors(idx)
        obs0 = self._getitem_transformed(self.data.pos_to_idx(f0))
        obs1 = self._getitem_transformed(self.data.pos_to_idx(f1))
        obs2 = self._getitem_transformed(self.data.pos_to_idx(f2))
        return obs0, obs1, obs2

    def sample_factors(self, idx):
        # SAMPLE FACTOR INDICES
        p_k, n_k = self._sample_num_factors()
        p_shared_indices, n_shared_indices = self._sample_shared_indices(p_k, n_k)
        # SAMPLE FACTORS - sample, resample and replace shared factors with originals
        anchor_factors = self.data.idx_to_pos(idx)
        positive_factors, negative_factors = self._resample_factors(anchor_factors)
        positive_factors[p_shared_indices] = anchor_factors[p_shared_indices]
        negative_factors[n_shared_indices] = negative_factors[n_shared_indices]
        # SWAP IF +VE FURTHER THAN -VE
        if self._swap_metric is not None:
            positive_factors, negative_factors = self._swap_factors(anchor_factors, positive_factors, negative_factors)
        return anchor_factors, positive_factors, negative_factors

    # --------------------------------------------------------------------- #
    # HELPER                                                                #
    # --------------------------------------------------------------------- #

    def _min_max_from_range(self, p_range, n_range, max_values, n_is_offset):
        p_min, p_max = normalise_range_pair(p_range, max_values)
        n_min, n_max = normalise_range_pair(n_range, max_values)
        # cross factor assertions
        if not n_is_offset:
            assert np.all(p_min < n_min)
            assert np.all(p_max <= n_max)
        else:
            assert np.all((p_max + n_min) <= max_values)
        return p_min, p_max, n_min, n_max

    def _sample_num_factors(self):
        p_k = np.random.randint(self.p_k_min, self.p_k_max + 1)
        # sample for negative
        if self.n_k_sample_mode == 'offset':
            n_k = np.random.randint(p_k + self.n_k_min, min(p_k + self.n_k_max, self.data.num_factors) + 1)
        elif self.n_k_sample_mode == 'normal':
            n_k = np.random.randint(max(p_k, self.n_k_min), self.n_k_max + 1)
        elif self.n_radius_sample_mode == 'unchecked':
            n_k = np.random.randint(self.n_k_min, self.n_k_max + 1)
        else:
            raise KeyError
        # we're done!
        return p_k, n_k

    def _resample_factors(self, anchor_factors):
        positive_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_min=self.p_radius_min, r_max=self.p_radius_max)
        # sample for negative
        if self.n_radius_sample_mode == 'offset':
            sampled_radius = np.abs(anchor_factors - positive_factors)
            n_radius_min = sampled_radius + self.n_radius_min
            n_radius_max = sampled_radius + self.n_radius_max
            negative_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_min=n_radius_min, r_max=n_radius_max)
        elif self.n_radius_sample_mode == 'normal':
            n_radius_min = np.minimum(np.abs(anchor_factors - positive_factors), self.n_radius_min)
            negative_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_min=n_radius_min, r_max=self.n_radius_max)
        elif self.n_radius_sample_mode == 'unchecked':
            negative_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_min=self.n_radius_min, r_max=self.n_radius_max)
        else:
            raise KeyError
        # we're done!
        return positive_factors, negative_factors

    def _sample_shared_indices(self, p_k, n_k):
        p_shared_indices = np.random.choice(self.data.num_factors, size=self.data.num_factors-p_k, replace=False)
        # sample for negative
        if self.n_k_is_shared:
            n_shared_indices = p_shared_indices[:self.data.num_factors-n_k]
        else:
            n_shared_indices = np.random.choice(self.data.num_factors, size=self.data.num_factors-n_k, replace=False)
        # we're done!
        return p_shared_indices, n_shared_indices

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


def sample_radius(value, low, high, r_min, r_max):
    """
    Sample around the given value (low <= value < high),
    the resampled value will lie in th same range.
    - sampling occurs in a radius around the value
    """
    value = np.array(value)
    assert np.all(low <= value)
    assert np.all(value < high)
    # sample for new value
    return randint2(
        a_low=np.maximum(value - r_max, low),
        a_high=value - r_min + 1,
        # if r_min == 0, then the ranges overlap, so we must shift one of them.
        b_low=value + r_min + (r_min == 0),
        b_high=np.minimum(value + r_max, high),
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
