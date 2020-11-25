import numpy as np
from disent.data.groundtruth.base import GroundTruthData
from disent.dataset.groundtruth import GroundTruthDataset


class GroundTruthDatasetOrigWeakPairs(GroundTruthDataset):

    def __init__(
            self,
            ground_truth_data: GroundTruthData,
            transform=None,
            augment=None,
            # num_differing_factors
            p_k: int = 1,
    ):
        """
        Dataset that emulates choosing factors like:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
        """
        super().__init__(ground_truth_data=ground_truth_data, transform=transform, augment=augment)
        # DIFFERING FACTORS
        self.p_k = p_k

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
        This function is based on _sample_weak_pair_factors()
        Except deterministic for the first item in the pair, based off of idx.
        """
        # randomly sample the first observation -- In our case we just use the idx
        sampled_factors = self.data.idx_to_pos(idx)
        # sample the next observation with k differing factors
        next_factors, k = _sample_k_differing(sampled_factors, self.data, k=self.p_k)
        # return the samples
        return sampled_factors, next_factors


def _sample_k_differing(factors, ground_truth_data: GroundTruthData, k=1):
    """
    Resample the factors used for the corresponding item in a pair.
      - Based on simple_dynamics() from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
    """
    # checks for factors
    factors = np.array(factors)
    assert factors.ndim == 1
    # sample k
    if k <= 0:
        k = np.random.randint(1, ground_truth_data.num_factors)
    # randomly choose 1 or k
    # TODO: This is in disentanglement lib, HOWEVER is this not a mistake?
    #       A bug report has been submitted to disentanglement_lib for clarity:
    #       https://github.com/google-research/disentanglement_lib/issues/31
    k = np.random.choice([1, k])
    # generate list of differing indices
    index_list = np.random.choice(len(factors), k, replace=False)
    # randomly update factors
    for index in index_list:
        factors[index] = np.random.choice(ground_truth_data.factor_sizes[index])
    # return!
    return factors, k


def _sample_weak_pair_factors(gt_data: GroundTruthData):
    """
    Sample a weakly supervised pair from the given GroundTruthData.
      - Based on weak_dataset_generator() from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
    """
    # randomly sample the first observation
    sampled_factors = gt_data.sample_factors(1)
    # sample the next observation with k differing factors
    next_factors, k = _sample_k_differing(sampled_factors, gt_data, k=1)
    # return the samples
    return sampled_factors, next_factors
