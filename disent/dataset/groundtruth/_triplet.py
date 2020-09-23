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


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
