import logging
import numpy as np
from disent.data.groundtruth.base import GroundTruthData
from disent.dataset.groundtruth import GroundTruthDatasetPairs

log = logging.getLogger(__name__)


# ========================================================================= #
# triplets                                                                  #
# ========================================================================= #


class GroundTruthDatasetTriples(GroundTruthDatasetPairs):

    def __init__(
            self,
            ground_truth_data: GroundTruthData,
            transform=None,
            k=1,
            n_k='uniform',
            resample_radius='inf',
            swap_if_wrong=False,
    ):
        super().__init__(
            ground_truth_data=ground_truth_data,
            transform=transform,
            k=k,
            resample_radius=resample_radius
        )

        # number of varied factors between pairs
        self._n_k = self.data.num_factors if (n_k is None) else n_k
        assert isinstance(self._n_k, str) or isinstance(self._n_k, int), f'n_k must be "uniform" or an integer k < n_k <= d, k={self._k}, d={self.data.num_factors}'
        if isinstance(self._n_k, int):
            assert self._k < self._n_k, f'n_k must be greater than k, k={self._k}'
            assert self._n_k <= self.data.num_factors, f'cannot vary more factors than there exist, k must be <= {self.data.num_factors}'

        # swap if the sampled factors are ordered wrong
        self._swap_metric = 'factors' if swap_if_wrong else None
        assert self._swap_metric in {None, 'factors', 'manhattan'}

    def sample_factors(self, idx):
        # get factors corresponding to index
        anchor_factors = self.data.idx_to_pos(idx)

        # get fixed or random k (k is number of factors that differ)
        p_k = np.random.randint(1, self.data.num_factors) if (self._k == 'uniform') else self._k
        n_k = np.random.randint(p_k+1, self.data.num_factors+1) if (self._n_k == 'uniform') else self._n_k

        positive_factors = self.data.resample_radius(anchor_factors, resample_radius=self._resample_radius, distinct=True, num_shared_factors=self.data.num_factors-p_k)
        negative_factors = self.data.resample_radius(anchor_factors, resample_radius=self._resample_radius, distinct=True, num_shared_factors=self.data.num_factors-n_k)

        # swap if number of shared factors is less for the positive
        if self._swap_metric is not None:
            pass  # do nothing!
        elif self._swap_metric == 'factors':
            # use the number of factors that have changed.
            if np.sum(anchor_factors == positive_factors) < np.sum(anchor_factors == negative_factors):
                positive_factors, negative_factors = negative_factors, positive_factors
                log.warning('Swapped factors based on number of factors')
        elif self._swap_metric == 'manhattan':
            # use manhattan distance along the factors
            if np.sum(np.abs(anchor_factors - positive_factors)) < np.sum(np.abs(anchor_factors - negative_factors)):
                positive_factors, negative_factors = negative_factors, positive_factors
                log.warning('Swapped factors based on Manhattan distance')
        else:
            raise KeyError

        # return observations
        return anchor_factors, positive_factors, negative_factors


class TripletFactorSampler(object):

    def __init__(
            self,
            num_factors: int,
            p_range=(1, -2),
            n_range=(1, -1),
            n_is_offset=True,
            share_p=True,
    ):
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        assert 1 <= num_factors
        self.num_factors = num_factors
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        p_range = np.array(p_range)
        n_range = np.array(n_range)
        # if not a 2 tuple, repeat. This fixes the min == max.
        if p_range.shape == (): p_range = p_range.repeat(2)
        if n_range.shape == (): n_range = n_range.repeat(2)
        # check final shape
        assert p_range.shape == (2,)
        assert n_range.shape == (2,)
        # compute the variability for each factor!
        p_range[p_range < 0] += num_factors + 1
        n_range[n_range < 0] += num_factors + 1
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.p_min, self.p_max = p_range
        self.n_min, self.n_max = n_range
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        # check that min <= max
        assert self.p_min <= self.p_max
        assert self.n_min <= self.n_max
        # check that everything is in the right range [1, -1]
        assert 0 <= self.p_min and self.p_max <= num_factors
        assert 0 <= self.n_min and self.n_max <= num_factors
        # cross factor assertions
        if not n_is_offset:
            assert self.p_min <= self.n_min
            assert self.p_max <= self.n_max
        else:
            assert (self.p_max + self.n_min) <= num_factors
        # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #
        self.n_is_offset = n_is_offset
        self.share_p = share_p

    def sample_num_differing(self):
        p_num = np.random.randint(self.p_min, self.p_max + 1)
        if self.n_is_offset:
            # Sample so that [n_min, n_max] is offset from p_num
            n_num = np.random.randint(p_num + self.n_min, min(p_num + self.n_max, self.num_factors) + 1)
        else:
            # Sample [p_min, p_max] and [n_min, n_max] individually
            n_num = np.random.randint(max(p_num, self.n_min), self.n_max + 1)
        return p_num, n_num

    def sample_shared_indices(self):
        p_k, n_k = self.sample_num_differing()
        p_num_shared = self.num_factors - p_k
        n_num_shared = self.num_factors - n_k
        # sample
        if self.share_p:
            p_shared_indices = np.random.choice(self.num_factors, size=p_num_shared, replace=False)
            n_shared_indices = p_shared_indices[:p_num_shared]
        else:
            p_shared_indices = np.random.choice(self.num_factors, size=p_num_shared, replace=False)
            n_shared_indices = np.random.choice(self.num_factors, size=n_num_shared, replace=False)
        # return
        return p_shared_indices, n_shared_indices


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
