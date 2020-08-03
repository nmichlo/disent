import logging
import numpy as np
from disent.dataset import GroundTruthDataset
from disent.dataset.pairs import PairedVariationDataset

log = logging.getLogger(__name__)

# ========================================================================= #
# triplets                                                                  #
# ========================================================================= #


class SupervisedTripletDataset(PairedVariationDataset):
    
    def __init__(self, dataset: GroundTruthDataset, k='uniform', variation_factor_indices=None, return_factors=False, mode='sample'):
        super().__init__(dataset, k=k, variation_factor_indices=variation_factor_indices, return_factors=return_factors)
        self.mode = mode

    def sample_factors(self, idx):
        # get factors corresponding to index
        anchor_factors = self._dataset.data.idx_to_pos(idx)

        # get fixed or random k (k is number of factors that differ)
        if self.mode == 'sample':
            # standard sampling operation for number of differing factors
            p_k = np.random.randint(1, self._num_variation_factors) if self._k == 'uniform' else self._k
            n_k = np.random.randint(p_k, self._num_variation_factors)
        elif self.mode == 'max_diff':
            raise KeyError(f'{self.mode=} has been disabled')
            # maximize divergence between positive and negative
            p_k = 1
            n_k = self._num_variation_factors - 1
        elif self.mode == 'min_diff':
            raise KeyError(f'{self.mode=} has been disabled')
            # minimize divergence between positive and negative
            p_k = 1
            n_k = 2
        else:
            raise KeyError(f'Invalid mode: {self.mode}')

        positive_factors = anchor_factors
        while np.all(anchor_factors == positive_factors):
            # these should generally differ by less factors
            # make k random indices not shared + resample paired item, differs by at most k factors of variation
            p_num_shared = self._dataset.data.num_factors - p_k
            p_shared_indices = np.random.choice(self._variation_factor_indices, size=p_num_shared, replace=False)
            positive_factors = self._dataset.data.resample_factors(anchor_factors[np.newaxis, :], p_shared_indices)[0]

        negative_factors = anchor_factors
        while np.all(anchor_factors == negative_factors):
            # these should generally differ by more factors
            # make k random indices not shared + resample paired item, differs by at most k factors of variation
            n_num_shared = self._dataset.data.num_factors - n_k
            n_shared_indices = np.random.choice(self._variation_factor_indices, size=n_num_shared, replace=False)
            negative_factors = self._dataset.data.resample_factors(anchor_factors[np.newaxis, :], n_shared_indices)[0]

        # swap if number of shared factors is less for the positive | This is not ideal
        # (swap if number of differing factors [k] is greater for the positive)
        if np.sum(anchor_factors == positive_factors) < np.sum(anchor_factors == negative_factors):
            positive_factors, negative_factors = negative_factors, positive_factors
            log.warning('Swapped factors')

        # CHECKED! choosing factors like this results in:
        # np.sum(anchor_factors == positive_factors) >= np.sum(anchor_factors == negative_factors)
        # ie. num positive shared factors >= num negative shared factors

        # return observations
        return anchor_factors, positive_factors, negative_factors
    

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
