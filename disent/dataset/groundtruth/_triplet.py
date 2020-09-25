import logging
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from disent.data.groundtruth import XYGridData
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
        # checks
        assert swap_metric in {None, 'factors', 'manhattan', 'manhattan_ratio', 'euclidean', 'euclidean_ratio'}
        assert n_k_sample_mode in {'offset', 'normal', 'unchecked'}
        assert n_radius_sample_mode in {'offset', 'normal', 'unchecked'}
        # DIFFERING FACTORS
        self.n_k_sample_mode = n_k_sample_mode
        self.n_k_is_shared = n_k_is_shared
        self.p_k_min, self.p_k_max, self.n_k_min, self.n_k_max = self._min_max_from_range(p_range=p_k_range, n_range=n_k_range, max_values=self.data.num_factors, n_sample_mode=n_k_sample_mode)
        # RADIUS SAMPLING
        self.n_radius_sample_mode = n_radius_sample_mode
        self.p_radius_min, self.p_radius_max, self.n_radius_min, self.n_radius_max = self._min_max_from_range(p_range=p_radius_range, n_range=n_radius_range, max_values=self.data.factor_sizes, n_sample_mode=n_radius_sample_mode)
        # SWAP: if negative is not further than the positive
        self._swap_metric = swap_metric

    # --------------------------------------------------------------------- #
    # CORE                                                                  #
    # --------------------------------------------------------------------- #

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
        negative_factors[n_shared_indices] = anchor_factors[n_shared_indices]
        # SWAP IF +VE FURTHER THAN -VE
        if self._swap_metric is not None:
            positive_factors, negative_factors = self._swap_factors(anchor_factors, positive_factors, negative_factors)
        return anchor_factors, positive_factors, negative_factors

    # --------------------------------------------------------------------- #
    # HELPER                                                                #
    # --------------------------------------------------------------------- #

    def _min_max_from_range(self, p_range, n_range, max_values, n_sample_mode):
        p_min, p_max = normalise_range_pair(p_range, max_values)
        n_min, n_max = normalise_range_pair(n_range, max_values)
        # cross factor assertions
        if n_sample_mode == 'offset':
            assert np.all((p_max + n_min) <= max_values)
        else:
            assert np.all(p_min <= n_min)
            assert np.all(p_max <= n_max)
        # we're done!
        return p_min, p_max, n_min, n_max

    def _sample_num_factors(self):
        p_k = np.random.randint(self.p_k_min, self.p_k_max + 1)
        # sample for negative
        if self.n_k_sample_mode == 'offset':
            n_k = np.random.randint(p_k + self.n_k_min, min(p_k + self.n_k_max, self.data.num_factors) + 1)
        elif self.n_k_sample_mode == 'normal':
            n_k = np.random.randint(max(p_k, self.n_k_min), self.n_k_max + 1)
        elif self.n_k_sample_mode == 'unchecked':
            n_k = np.random.randint(self.n_k_min, self.n_k_max + 1)
        else:
            raise KeyError(f'Unknown mode: {self.n_k_sample_mode=}')
        # we're done!
        return p_k, n_k

    def _resample_factors(self, anchor_factors):
        positive_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_min=self.p_radius_min, r_max=self.p_radius_max)
        # sample for negative
        if self.n_radius_sample_mode == 'offset':
            sampled_radius = np.abs(anchor_factors - positive_factors)
            n_radius_min = sampled_radius + self.n_radius_min
            n_radius_max = self.n_radius_max # sampled_radius + self.n_radius_max
            print(n_radius_min, self.n_radius_min, n_radius_max, self.n_radius_max, self.data.factor_sizes)
            negative_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_min=n_radius_min, r_max=n_radius_max)
        elif self.n_radius_sample_mode == 'normal':
            n_radius_min = np.maximum(np.abs(anchor_factors - positive_factors), self.n_radius_min)
            print(n_radius_min, self.n_radius_min)
            negative_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_min=n_radius_min, r_max=self.n_radius_max)
        elif self.n_radius_sample_mode == 'unchecked':
            negative_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_min=self.n_radius_min, r_max=self.n_radius_max)
        else:
            raise KeyError(f'Unknown mode: {self.n_radius_sample_mode=}')
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

    # --------------------------------------------------------------------- #
    # END CLASS                                                             #
    # --------------------------------------------------------------------- #


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
    # print('randint2', a_low, a_high, b_low, b_high)
    # convert
    print(a_low, a_high, b_low, b_high)
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
    # print('sample_radius', value, low, high, r_min, r_max)
    value = np.array(value)
    assert np.all(low <= value)
    assert np.all(value < high)
    print(value, low, high, r_min, r_max)
    # sample for new value
    return randint2(
        a_low=np.maximum(value - r_max, low),
        a_high=value - r_min + 1,
        # if r_min == 0, then the ranges overlap, so we must shift one of them.
        b_low=value + r_min + (r_min == 0),
        b_high=np.minimum(value + r_max, high) + 1,
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

if __name__ == '__main__':

    dataset = GroundTruthDatasetTriples(
        XYGridData(),
        # factor sampling
        p_k_range=(1, 1),
        n_k_range=(2, 2),
        n_k_sample_mode='unchecked',
        n_k_is_shared=True,
        # radius sampling
        p_radius_range=(1, 1),
        n_radius_range=(1, -1),
        n_radius_sample_mode='offset',
        # final checks
        swap_metric=None,
    )

    print(dataset.data.factor_sizes)

    stats = defaultdict(int)
    N = min(len(dataset), 2000)

    for i in tqdm(range(N)):
        # CHECK DIFFERENT FACTORS
        p_k, n_k = dataset._sample_num_factors()
        stats['p_k'] += p_k
        stats['n_k'] += n_k
        # CHECK ALL SAMPLING
        a, p, n = dataset.sample_factors(i)
        print(a, p, n, '|', np.abs(a - p), np.abs(a - n), np.abs(p - n))
        stats['a_p_diffs'] += np.sum(a != p)
        stats['a_n_diffs'] += np.sum(a != n)
        stats['a_p_ave_dist'] += np.abs(a - p).sum()
        stats['a_n_ave_dist'] += np.abs(a - n).sum()
        stats['p_n_ave_dist'] += np.abs(p - n).sum()

    for k, v in stats.items():
        print(k, v / N)

    # samples = []
    # for i in range(10000):
    #     sample = sample_radius(2, 0, 5, 1, 1)
    #     samples.append(sample)
    # samples = np.array(samples)
    #
    # print(np.unique(samples))