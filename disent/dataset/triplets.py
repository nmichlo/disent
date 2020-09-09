import logging
import numpy as np
from disent.dataset import GroundTruthDataset
from disent.dataset.pairs import PairedVariationDataset

log = logging.getLogger(__name__)

# ========================================================================= #
# triplets                                                                  #
# ========================================================================= #


class SupervisedTripletDataset(PairedVariationDataset):

    def __init__(
            self,
            dataset: GroundTruthDataset,
            k=1,
            n_k='uniform',
            swap_if_wrong=False,
            force_different_factors=True,
            variation_factor_indices=None,
            return_factors=False,
            resample_radius='inf',
            random_copy_chance=0,
            random_transform=None,
    ):
        super().__init__(
            dataset,
            k=k,
            force_different_factors=force_different_factors,
            variation_factor_indices=variation_factor_indices,
            return_factors=return_factors,
            resample_radius=resample_radius,
            random_copy_chance=random_copy_chance,
            random_transform=random_transform
        )
        self.swap_if_wrong = swap_if_wrong
        # number of varied factors between pairs
        self._n_k = self._num_variation_factors if (n_k is None) else n_k
        # verify k
        assert isinstance(self._n_k, str) or isinstance(self._n_k, int), f'n_k must be "uniform" or an integer k < n_k <= d, k={self._k}, d={self._num_variation_factors}'
        if isinstance(self._n_k, int):
            assert self._k < self._n_k, f'n_k must be greater than k, k={self._k}'
            assert self._n_k <= self._num_variation_factors, f'cannot vary more factors than there exist, k must be <= {self._num_variation_factors}'

    def sample_factors(self, idx):
        # get factors corresponding to index
        anchor_factors = self._dataset.data.idx_to_pos(idx)

        # get fixed or random k (k is number of factors that differ)
        p_k = np.random.randint(1, self._num_variation_factors) if (self._k == 'uniform') else self._k
        n_k = np.random.randint(p_k+1, self._num_variation_factors+1) if (self._n_k == 'uniform') else self._n_k

        positive_factors = self._resample_factors(anchor_factors, p_k)
        negative_factors = self._resample_factors(anchor_factors, n_k)

        if self.swap_if_wrong:
            # swap if number of shared factors is less for the positive
            if True:  # self.resample_radius is None:
                # use the number of factors that have changed.
                if np.sum(anchor_factors == positive_factors) < np.sum(anchor_factors == negative_factors):
                    positive_factors, negative_factors = negative_factors, positive_factors
                    log.warning('Swapped factors based on number of factors')
            else:
                # TODO: enable this functionality
                # use manhattan distance along the factors
                if np.sum(np.abs(anchor_factors - positive_factors)) < np.sum(np.abs(anchor_factors - negative_factors)):
                    positive_factors, negative_factors = negative_factors, positive_factors
                    log.warning('Swapped factors based on Manhattan distance')

        # return observations
        return anchor_factors, positive_factors, negative_factors


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
