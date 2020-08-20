import numpy as np
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from disent.dataset.ground_truth_data.base_data import GroundTruthData
from disent.dataset.ground_truth_data.data_xygrid import XYGridData
from disent.dataset.single import GroundTruthDataset


# ========================================================================= #
# pairs                                                                     #
# ========================================================================= #


class RandomPairDataset(Dataset):

    def __init__(self, dataset: Dataset):
        assert len(dataset) > 1, 'Dataset must be contain more than one observation.'
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # find differing random index, nearly always this will only run once.
        rand_idx, attempts = idx, 0
        while rand_idx == idx:
            rand_idx = np.random.randint(len(self.dataset))
            attempts += 1
            if attempts > 1000:
                # pretty much impossible unless your dataset is of size 1, or your prng is broken...
                raise IndexError('Unable to find random index that differs.')
        # return elements
        return (self.dataset[idx], idx), (self.dataset[rand_idx], rand_idx)


class PairedVariationDataset(Dataset):

    def __init__(
            self,
            dataset: GroundTruthDataset,
            k=1,
            force_different_factors=True,
            variation_factor_indices=None,
            return_factors=False,
            sample_nearby=False,
            random_copy_chance=0,
            random_transform=None,
    ):
        """
        Dataset that pairs together samples with at most k differing factors of variation.

        dataset: A dataset that extends GroundTruthData
        k: An integer (k), None (k=d-1), or "uniform" (random k in range 1 to d-1)
        variation_factor_indices: The indices of the factors of variation that are sampled between pairs, if None (all factors are sampled)
        """
        assert isinstance(dataset, GroundTruthDataset), 'passed object is not an instance of GroundTruthDataset'
        assert len(dataset) > 1, 'Dataset must be contain more than one observation.'
        # wrapped dataset
        self._dataset = dataset
        # possible fixed dimensions between pairs
        self._variation_factor_indices = np.arange(self._dataset.data.num_factors) if (variation_factor_indices is None) else np.array(variation_factor_indices)
        self._variation_factor_sizes = np.array(self._dataset.data.factor_sizes)[self._variation_factor_indices]
        # d
        self._num_variation_factors = len(self._variation_factor_indices)
        # number of varied factors between pairs
        self._k = self._num_variation_factors - 1 if (k is None) else k
        # verify k
        assert isinstance(self._k, str) or isinstance(self._k, int), f'k must be "uniform" or an integer 1 <= k <= d-1, d={self._num_variation_factors}'
        if isinstance(self._k, int):
            assert 1 <= self._k, 'k cannot be less than 1'
            assert self._k < self._num_variation_factors, f'all factors cannot be varied for each pair, k must be less than {self._num_variation_factors}'
        # if we must return (x, y) instead of just x, where y is the factors for x.
        self._return_factors = return_factors
        # if sampled factors MUST be different
        self.force_different_factors = force_different_factors
        # if we must sample according to offsets, rather than along an entire axis
        # TODO: change this to a radius
        self._sample_nearby = sample_nearby
        # randomness
        assert random_copy_chance >= 0, f'{random_copy_chance=} must be >= 0'
        self._random_copy_chance = random_copy_chance
        self._random_transform = random_transform

    def __len__(self):
        # TODO: is dataset as big as the latent space OR as big as the orig.
        # return self._latent_space.size
        return len(self._dataset.data)

    def __getitem__(self, idx):
        # get random factor pairs
        factors = list(self.sample_factors(idx))
        assert 2 <= len(factors) <= 3, 'More factors are not yet supported!'
        # transform if needed, or randomly replace!
        if self._random_copy_chance > 0:
            if np.random.random_sample() < self._random_copy_chance:
                factors[1] = factors[0]  # we still want the negatives in triples to be further away
        # get observations from factors
        observations = [self._dataset[self._dataset.data.pos_to_idx(pos)] for pos in factors]
        # do random transformations
        if self._random_transform:
            observations = [self._random_transform(x) for x in observations]
        # return observations and factors, or just observations
        if self._return_factors:
            return list(zip(observations, factors))
        else:
            return observations

    def sample_factors(self, idx):
        """
        Excerpt from Weakly-Supervised Disentanglement Without Compromises:
        [section 5. Experimental results]

        CREATE DATA SETS: with weak supervision from the existing
        disentanglement data sets:
        1. we first sample from the discrete z according to the ground-truth generative model (1)–(2).
        2. Then, we sample k factors of variation that should not be shared by the two images and re-sample those coordinates to obtain z˜.
           This ensures that each image pair differs in at most k factors of variation.

        For k we consider the range from 1 to d − 1.
        This last setting corresponds to the case where all but one factor of variation are re-sampled.

        We study both the case where k is constant across all pairs in the data set and where k is sampled uniformly in the range [d − 1] for every training pair (k = Rnd in the following).
        Unless specified otherwise, we aggregate the results for all values of k.
        """

        # get factors corresponding to index
        orig_factors = self._dataset.data.idx_to_pos(idx)
        # get fixed or random k (k is number of factors that differ)
        k = np.random.randint(1, self._num_variation_factors) if (self._k == 'uniform') else self._k
        # return observations
        return orig_factors, self._resample_factors(orig_factors, k)

    def _resample_factors(self, base_factors, k):
        resampled_factors = None
        while (resampled_factors is None) or np.all(base_factors == resampled_factors):
            # make k random indices not shared + resample paired item, differs by at most k factors of variation
            num_shared = self._dataset.data.num_factors - k
            shared_indices = np.random.choice(self._variation_factor_indices, size=num_shared, replace=False)
            # how the non-shared indices are to be sampled
            if self._sample_nearby:
                share_mask = np.zeros(self._num_variation_factors, dtype='bool')
                share_mask[shared_indices] = True
                offsets = (np.random.randint(0, 2, size=self._num_variation_factors) * 2 - 1) * (~share_mask)
                resampled_factors = np.clip(base_factors + offsets, 0, self._variation_factor_sizes - 1)  # CYCLIC: (base_factors + offsets) % self._variation_factor_sizes
            else:
                resampled_factors = self._dataset.data.resample_factors(base_factors[np.newaxis, :], shared_indices)[0]
            # dont retry if sampled factors are the same
            if not self.force_different_factors:
                break
        return resampled_factors


class PairedContrastiveDataset(Dataset):

    def __init__(self, dataset: GroundTruthDataset, transforms):
        """
        Dataset that creates a randomly transformed contrastive pair.

        dataset: A dataset that extends GroundTruthData
        transforms: transform to apply - should make use of random transforms.
        """
        assert isinstance(dataset, GroundTruthDataset), 'passed object is not an instance of GroundTruthDataset'
        assert len(dataset) > 1, 'Dataset must be contain more than one observation.'
        # wrapped dataset
        self._dataset = dataset
        # random transforms
        self._transforms = transforms

    def __len__(self):
        return len(self._dataset.data)

    def __getitem__(self, idx):
        x0 = self._dataset[idx]
        x1 = self._transforms(x0)
        return x0, x1


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
