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
            augment=None,
            # factor sampling
            p_k_range=(0, -1),
            n_k_range=(0, -1),
            n_k_sample_mode='bounded_below',
            n_k_is_shared=True,
            # radius sampling
            p_radius_range=(0, -1),
            n_radius_range=(0, -1),
            n_radius_sample_mode='bounded_below',
            # final checks
            swap_metric=None,
            swap_chance=None,
    ):
        super().__init__(ground_truth_data=ground_truth_data, transform=transform, augment=augment)
        # checks
        assert swap_metric in {None, 'k', 'manhattan', 'manhattan_norm', 'euclidean', 'euclidean_norm'}, f'Invalid {swap_metric=}'
        assert n_k_sample_mode in {'offset', 'bounded_below', 'random'}, f'Invalid {n_k_sample_mode=}'
        assert n_radius_sample_mode in {'offset', 'bounded_below', 'random'}, f'Invalid {n_radius_sample_mode=}'
        # DIFFERING FACTORS
        self.n_k_sample_mode = n_k_sample_mode
        self.n_k_is_shared = n_k_is_shared
        self.p_k_min, self.p_k_max, self.n_k_min, self.n_k_max = self._min_max_from_range(
            p_range=p_k_range,
            n_range=n_k_range,
            max_values=self.data.num_factors,
            n_sample_mode=n_k_sample_mode,
            is_radius=False
        )
        # RADIUS SAMPLING
        self.n_radius_sample_mode = n_radius_sample_mode
        self.p_radius_min, self.p_radius_max, self.n_radius_min, self.n_radius_max = self._min_max_from_range(
            p_range=p_radius_range,
            n_range=n_radius_range,
            max_values=self.data.factor_sizes,
            n_sample_mode=n_radius_sample_mode,
            is_radius=True
        )
        # SWAP: if negative is not further than the positive
        self._swap_metric = swap_metric
        self._swap_chance = swap_chance
        if swap_chance is not None:
            assert 0 <= swap_chance <= 1, f'{swap_chance=} must be in range 0 to 1.'

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # CORE                                                                  #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def __getitem__(self, idx):
        f0, f1, f2 = self.datapoint_sample_factors_triplet(idx)
        return self.dataset_get_observation(
            self.data.pos_to_idx(f0),
            self.data.pos_to_idx(f1),
            self.data.pos_to_idx(f2),
        )

    def datapoint_sample_factors_triplet(self, idx):
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
        # RANDOMLY SWAP +ve AND -ve IF CHANCE:
        if self._swap_chance is not None:
            if np.random.random() < self._swap_chance:
                positive_factors, negative_factors = negative_factors, positive_factors
        # return factors!
        return anchor_factors, positive_factors, negative_factors

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # HELPER                                                                #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _min_max_from_range(self, p_range, n_range, max_values, n_sample_mode, is_radius=False):
        p_min, p_max = normalise_range_pair(p_range, max_values)
        n_min, n_max = normalise_range_pair(n_range, max_values)
        # cross factor assertions
        if n_sample_mode == 'offset':
            if is_radius:
                # TODO: radius for -ve values should be handled when sampling, not here.
                if not np.all((p_max + n_min) <= np.floor_divide(max_values, 2)):
                    raise FactorSizeError(f'Factor dimensions are too small for given offset radius:'
                                          f'\n\tUnsatisfied: p_max + offset_min <= max_size // 2'
                                          f'\n\tUnsatisfied: {p_max} + {n_min} <= {np.array(max_values)} // 2'
                                          f'\n\tUnsatisfied: {p_max + n_min} <= {np.floor_divide(max_values, 2)}')
            else:
                if not np.all((p_max + n_min) <= max_values):
                    raise FactorSizeError(f'Factor dimensions are too small for given offset range:'
                                          f'\n\tUnsatisfied: p_max + offset_min <= max_size'
                                          f'\n\tUnsatisfied: {p_max} + {n_min} <= {np.array(max_values)}')
        elif n_sample_mode == 'bounded_below':
            if not (np.all(p_max <= n_max)):
                raise FactorSizeError(f'Ranges are not staggered.'
                                      f'\n\tUnsatisfied: p_max <= n_max'
                                      f'\n\tUnsatisfied: {p_max} <= {n_max}')
            if not (np.all(p_max <= max_values) and np.all(n_max <= max_values)):
                raise FactorSizeError('Factor dimensions are too small for given range:'
                                      f'\n\tUnsatisfied: p_max <= max_size and n_max <= max_size'
                                      f'\n\tUnsatisfied: {p_max} <= {np.array(max_values)} and {n_max} <= {np.array(max_values)}')
        else:
            if not (np.all(p_min <= n_min) and np.all(p_max <= n_max)):
                raise FactorSizeError(f'Ranges are not staggered. Should be p <= n:'
                                      f'\n\tUnsatisfied: p_min <= n_min and p_max <= n_max'
                                      f'\n\tUnsatisfied: {p_min} <= {n_min} and {p_max} <= {n_max}')
            if not (np.all(p_max <= max_values) and np.all(n_max <= max_values)):
                raise FactorSizeError('Factor dimensions are too small for given range:'
                                      f'\n\tUnsatisfied: p_max <= max_size and n_max <= max_size'
                                      f'\n\tUnsatisfied: {p_max} <= {np.array(max_values)} and {n_max} <= {np.array(max_values)}')
        # we're done!
        return p_min, p_max, n_min, n_max

    def _sample_num_factors(self):
        p_k = np.random.randint(self.p_k_min, self.p_k_max + 1)
        # sample for negative
        if self.n_k_sample_mode == 'offset':
            n_k = np.random.randint(p_k + self.n_k_min, min(p_k + self.n_k_max, self.data.num_factors) + 1)
        elif self.n_k_sample_mode == 'bounded_below':
            n_k = np.random.randint(max(p_k, self.n_k_min), self.n_k_max + 1)
        elif self.n_k_sample_mode == 'random':
            n_k = np.random.randint(self.n_k_min, self.n_k_max + 1)
        else:
            raise KeyError(f'Unknown mode: {self.n_k_sample_mode=}')
        # we're done!
        return p_k, n_k

    def _sample_shared_indices(self, p_k, n_k):
        p_shared_indices = np.random.choice(self.data.num_factors, size=self.data.num_factors-p_k, replace=False)
        # sample for negative
        if self.n_k_is_shared:
            n_shared_indices = p_shared_indices[:self.data.num_factors-n_k]
        else:
            n_shared_indices = np.random.choice(self.data.num_factors, size=self.data.num_factors-n_k, replace=False)
        # we're done!
        return p_shared_indices, n_shared_indices

    def _resample_factors(self, anchor_factors):
        # sample positive
        positive_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_low=self.p_radius_min, r_high=self.p_radius_max + 1)
        # negative arguments
        if self.n_radius_sample_mode == 'offset':
            sampled_radius = np.abs(anchor_factors - positive_factors)
            n_r_low = sampled_radius + self.n_radius_min
            n_r_high = sampled_radius + self.n_radius_max + 1
        elif self.n_radius_sample_mode == 'bounded_below':
            sampled_radius = np.abs(anchor_factors - positive_factors)
            n_r_low = np.maximum(sampled_radius, self.n_radius_min)
            n_r_high = self.n_radius_max + 1
        elif self.n_radius_sample_mode == 'random':
            n_r_low = self.n_radius_min
            n_r_high = self.n_radius_max + 1
        else:
            raise KeyError(f'Unknown mode: {self.n_radius_sample_mode=}')
        # sample negative
        negative_factors = sample_radius(anchor_factors, low=0, high=self.data.factor_sizes, r_low=n_r_low, r_high=n_r_high)
        # we're done!
        return positive_factors, negative_factors

    def _swap_factors(self, anchor_factors, positive_factors, negative_factors):
        if self._swap_metric == 'k':
            p_dist = np.sum(anchor_factors == positive_factors)
            n_dist = np.sum(anchor_factors == negative_factors)
        elif self._swap_metric == 'manhattan':
            p_dist = np.sum(np.abs(anchor_factors - positive_factors))
            n_dist = np.sum(np.abs(anchor_factors - negative_factors))
        elif self._swap_metric == 'manhattan_norm':
            p_dist = np.sum(np.abs((anchor_factors - positive_factors) / np.subtract(self.data.factor_sizes, 1)))
            n_dist = np.sum(np.abs((anchor_factors - negative_factors) / np.subtract(self.data.factor_sizes, 1)))
        elif self._swap_metric == 'euclidean':
            p_dist = np.linalg.norm(anchor_factors - positive_factors)
            n_dist = np.linalg.norm(anchor_factors - negative_factors)
        elif self._swap_metric == 'euclidean_norm':
            p_dist = np.linalg.norm((anchor_factors - positive_factors) / np.subtract(self.data.factor_sizes, 1))
            n_dist = np.linalg.norm((anchor_factors - negative_factors) / np.subtract(self.data.factor_sizes, 1))
        else:
            raise KeyError
        # perform swap
        if n_dist < p_dist:
            positive_factors, negative_factors = negative_factors, positive_factors
            # log.warning(f'Swapped factors based on metric: {self._swap_metric}')
        # return factors
        return positive_factors, negative_factors

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # END CLASS                                                             #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


class FactorSizeError(Exception):
    pass


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


# TODO: these functions should be moved into the state_space or at least a helper class!


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
    assert np.all(a_low <= a_high), f'a_low <= a_high | {a_low} <= {a_high}'
    assert np.all(b_low <= b_high), f'b_low <= b_high | {b_low} <= {b_high}'
    assert np.all(a_high <= b_low), f'a_high <= b_low | {a_high} <= {b_low}'
    # compute
    da = a_high - a_low
    db = b_high - b_low
    d = da + db
    # sampled
    offset = np.random.randint(0, d, size=size)
    offset += (da <= offset) * (b_low - a_high)
    return a_low + offset


def sample_radius(value, low, high, r_low, r_high):
    """
    Sample around the given value (low <= value < high),
    the resampled value will lie in th same range.
    - sampling occurs in a radius around the value
      r_low <= radius < r_high
    """
    value = np.array(value)
    assert np.all(low <= value)
    assert np.all(value < high)
    # sample for new value
    return randint2(
        a_low=np.maximum(value - r_high + 1, low),
        a_high=np.maximum(value - r_low + 1, low),
        # if r_min == 0, then the ranges overlap, so we must shift one of them.
        b_low=np.minimum(value + r_low + (r_low == 0), high),
        b_high=np.minimum(value + r_high, high),
    )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #

# if __name__ == '__main__':

    # def conf(data, ci=0.95, two_tailed=True):
    #     import scipy.stats
    #     ci = ((1 - ci) / 2) if two_tailed else (1 - ci)
    #     return scipy.stats.t.ppf(1 - ci, N - 1) * np.std(data) / np.sqrt(N)
    #
    # def conf_interval(data, ci=0.95):
    #     mean = np.mean(data)
    #     c = conf(data, ci, two_tailed=True)
    #     return np.array([mean - c, mean + c])
    #
    # dataset = GroundTruthDatasetTriples(
    #     XYGridData(),
    #     # factor sampling
    #     p_k_range=(1, 1),
    #     n_k_range=(1, 1),
    #     n_k_sample_mode='random',
    #     n_k_is_shared=True,
    #     # radius sampling
    #     p_radius_range=(1, 1),
    #     n_radius_range=(1, -1),
    #     n_radius_sample_mode='offset',
    #     # final checks
    #     swap_metric=None,
    # )
    #
    # print(dataset.data.factor_sizes)
    # print()
    #
    # stats = defaultdict(list)
    # N = min(len(dataset), 10_000)
    #
    # for i in tqdm(range(N)):
    #     # CHECK DIFFERENT FACTORS
    #     p_k, n_k = dataset._sample_num_factors()
    #     stats['p_k'].append(p_k)
    #     stats['n_k'].append(n_k)
    #     # CHECK ALL SAMPLING
    #     a, p, n = dataset.sample_factors(i)
    #     # print(f'a:{a} p:{p} n:{n} | ap:{np.abs(a - p)} an:{np.abs(a - n)} | pn:{np.abs(p - n)}')
    #     stats['a_p_diffs'].append(np.sum(a != p))
    #     stats['a_n_diffs'].append(np.sum(a != n))
    #     stats['a_p_ave_dist'].append(np.abs(a - p).sum())
    #     stats['a_n_ave_dist'].append(np.abs(a - n).sum())
    #     stats['p_n_ave_dist'].append(np.abs(p - n).sum())
    #
    # print(dataset.data.factor_sizes)
    # for k, v in stats.items():
    #     print(f'{k}: {np.around(np.mean(v), 3)} ± {np.around(conf(v, 0.95), 3)}')
    #
    # # check that resample radius is working correctly!
    # dataset = GroundTruthDatasetTriples(
    #     XYMultiGridData(1, 4),
    #     # factor sampling
    #     p_k_range=(1, 1),
    #     n_k_range=(1, 1),
    #     n_k_sample_mode='random',
    #     n_k_is_shared=True,
    #     # radius sampling
    #     p_radius_range=(1, 1),
    #     n_radius_range=(1, 1),
    #     n_radius_sample_mode='offset',
    #     # final checks
    #     swap_metric=None,
    # )
    #
    # for pair in dataset:
    #     obs0, _, obs1 = np.array(pair[0], dtype='int'), np.array(pair[1], dtype='int'), np.array(pair[2], dtype='int')
    #     # CHECKS
    #     diff = np.abs(obs1 - obs0)
    #     diff_coords = np.array(np.where(diff > 0)).T
    #     assert len(diff_coords) == 2  # check max changes
    #     dist = np.abs(diff_coords[0] - diff_coords[1])
    #     assert np.sum(dist > 0) == 1  # check max changes
    #     assert np.max(dist) == 2      # check radius
    #     # INFO
    #     print(concat_lines(*[((obs > 0) * [1, 2, 4]).sum(axis=-1) for obs in (obs0, obs1)]), '\n')

    # import time
    # t = time.time_ns()
    # N = 10000
    # samples = []
    # for i in range(N):
    #     sample = sample_radius(3, 0, 7, 0, 4)
    #     samples.append(sample)
    # samples = np.array(samples)
    #
    # u, c = np.unique(samples, return_counts=True)
    # idxs = np.argsort(u)
    # print(u[idxs])
    # print(c[idxs] / N)
    # ts = (time.time_ns() - t) / 1000_000_000
    # print(ts * 1000, 'ms')
    # print(N / ts, 'per/s')
