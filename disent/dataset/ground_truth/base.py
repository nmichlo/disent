import os
from typing import Tuple, Union
import h5py
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
import numpy as np
from disent.dataset.util.io import basename_from_url, download_file, ensure_dir_exists


# ========================================================================= #
# index                                                                     #
# ========================================================================= #


class DiscreteStateSpace(object):
    """
    State space where an index corresponds to coordinates in the factor space.
    ie. State space with multiple factors of variation, where each factor can be a different size.
    Heavily modified FROM: https://github.com/google-research/disentanglement_lib/blob/adb2772b599ea55c60d58fd4b47dff700ef9233b/disentanglement_lib/data/ground_truth/util.py
    """

    def __init__(self, factor_sizes):
        super().__init__()
        # dimension
        self._factor_sizes = np.array(factor_sizes)
        self._size = np.prod(factor_sizes)
        # dimension sampling
        self._factor_indices_set = set(range(self.num_factors))
        # helper data for conversion between factors and indexes
        bases = np.prod(self._factor_sizes) // np.cumprod([1, *self._factor_sizes])
        self._factor_divisors = bases[1:]
        self._factor_modulus = bases[:-1]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.idx_to_pos(idx)

    @property
    def size(self):
        return self._size

    @property
    def num_factors(self):
        return len(self._factor_sizes)

    def pos_to_idx(self, pos):
        return np.dot(pos, self._factor_divisors)

    def idx_to_pos(self, idx):
        return (idx % self._factor_modulus) // self._factor_divisors

    def sample_factors(self, num_samples, factor_indices=None):
        """
        sample randomly from all factors, otherwise the given factor_indices.
        returned values appear in the same order as factor_indices.
        """
        sizes = self._factor_sizes if (factor_indices is None) else self._factor_sizes[factor_indices]
        return np.random.randint(
            sizes,
            size=(num_samples, len(sizes))
        )

    # def sample_indices(self, num_samples):
    #     """Like sample_factors but returns indices."""
    #     return self.pos_to_idx(self.sample_factors(num_samples))

    def sample_missing_factors(self, values, factor_indices):
        """
        Samples the remaining factors not given in the dimension_indices.
        ie. fills in the missing values by sampling from the unused dimensions.
        returned values are ordered by increasing factor index and not factor_indices.
        """
        num_samples, num_dims = values.shape
        used_indices_set = set(factor_indices)
        # assertions
        assert num_dims == len(factor_indices), 'dimension count mismatch'
        assert len(used_indices_set) == len(factor_indices), 'dimension indices are duplicated'
        # set used dimensions
        all_values = np.zeros(shape=(num_samples, self.num_factors), dtype=np.int64)
        all_values[:, factor_indices] = values
        # sample for missing
        missing_indices = list(self._factor_indices_set - used_indices_set)
        all_values[:, missing_indices] = self.sample_factors(num_samples=num_samples, factor_indices=missing_indices)
        # return
        return all_values

    # def sample_missing_indices(self, values, factor_indices):
    #     """Like sample_missing_factors but returns indices."""
    #     return self.pos_to_idx(self.sample_missing_factors(values, factor_indices))

    def resampled_factors(self, values, factors_indices):
        """
        Resample across all the factors, keeping factor_indices constant.
        returned values are ordered by increasing factor index and not factor_indices.
        """
        return self.sample_missing_factors(values[:, factors_indices], factors_indices)

    # def resampled_indices(self, values, factors_indices):
    #     """Like resampled_factors but returns indices."""
    #     return self.pos_to_idx(self.resampled_factors(values, factors_indices))


# ========================================================================= #
# ground truth data                                                         #
# ========================================================================= #


class GroundTruthData(DiscreteStateSpace):

    def __init__(self):
        assert len(self.factor_names) == len(self.factor_sizes), 'Dimensionality mismatch of FACTOR_NAMES and FACTOR_DIMS'
        super().__init__(self.factor_sizes)

    def sample_observations(self, num_samples) -> list:
        """Sample a batch of observations X."""
        return self.sample(num_samples)[1]

    def sample_observations_from_factors(self, factors):
        """Sample a batch of observations X given a batch of factors Y."""
        indices = self.pos_to_idx(factors)
        observations = [self[idx] for idx in indices]
        return observations

    def sample(self, num_samples):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num_samples)
        observations = self.sample_observations_from_factors(factors)
        return factors, observations

    @property
    def factor_names(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    def __getitem__(self, indices):
        raise NotImplementedError


# ========================================================================= #
# Convert ground truth data to a dataset                                    #
# ========================================================================= #


class GroundTruthDataset(Dataset, GroundTruthData):
    """
    Converts ground truth data into a dataset
    """

    def __init__(self, ground_truth_data, transform=None):
        self.data = ground_truth_data
        # transform observation
        self.transform = transform
        # initialise GroundTruthData
        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, indices):
        """
        should return a single observation if an integer index, or
        an array of observations if indices is an array.
        """
        if torch.is_tensor(indices):
            indices = indices.tolist()

        image = self.data[indices]

        # https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py
        # PIL Image so that this is consistent with other datasets
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self.data.factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self.data.factor_sizes

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self.data.observation_shape

# ========================================================================= #
# paired factor of variation data                                        #
# ========================================================================= #


class DownloadableGroundTruthData(GroundTruthData):

    def __init__(self, data_dir='data', force_download=False):
        super().__init__()
        # paths
        self._data_dir = ensure_dir_exists(data_dir)
        self._data_path = os.path.join(self._data_dir, basename_from_url(self.dataset_url))
        # meta
        self._force_download = force_download
        # DOWNLOAD
        self._do_download_dataset()

    def _do_download_dataset(self):
        no_data = not os.path.exists(self._data_path)
        # download data
        if self._force_download or no_data:
            download_file(self.dataset_url, self._data_path)

    @property
    def dataset_path(self):
        '''path that the data should be loaded from in the child class'''
        return self._data_path

    @property
    def dataset_url(self) -> str:
        raise NotImplementedError()


class PreprocessedDownloadableGroundTruthData(DownloadableGroundTruthData):

    def __init__(self, data_dir='data', force_download=False, force_preprocess=False):
        super().__init__(data_dir=data_dir, force_download=force_download)
        # paths
        self._proc_path = f'{self._data_path}.processed'
        self._force_preprocess = force_preprocess
        # PROCESS
        self._do_download_and_process_dataset()

    def _do_download_dataset(self):
        # we skip this in favour of our new method,
        # so that we can lazily download the data.
        pass

    def _do_download_and_process_dataset(self):
        no_data = not os.path.exists(self._data_path)
        no_proc = not os.path.exists(self._proc_path)

        # preprocess only if required
        do_proc = self._force_preprocess or no_proc
        # lazily download if required for preprocessing
        do_data = self._force_download or (no_data and do_proc)

        if do_data:
            download_file(self.dataset_url, self._data_path)

        if do_proc:
            # TODO: also used in io save file, convert to with syntax.
            # save to a temporary location in case there is an error, we then know one occured.
            path_dir, path_base = os.path.split(self._proc_path)
            ensure_dir_exists(path_dir)
            temp_proc_path = os.path.join(path_dir, f'.{path_base}.temp')

            # process stuff
            self._preprocess_dataset(path_src=self._data_path, path_dst=temp_proc_path)

            # delete existing file if needed
            if os.path.isfile(self._proc_path):
                os.remove(self._proc_path)
            # move processed file to correct place
            os.rename(temp_proc_path, self._proc_path)

            assert os.path.exists(self._proc_path), f'Overridden _preprocess_dataset method did not initialise the required dataset file: dataset_path="{self._proc_path}"'

    @property
    def dataset_path(self):
        '''path that the dataset should be loaded from in the child class'''
        return self._proc_path

    def _preprocess_dataset(self, path_src, path_dst):
        raise NotImplementedError()


class Hdf5PreprocessedGroundTruthData(PreprocessedDownloadableGroundTruthData):
    """
    Automatically download and pre-process an hdf5 dataset into the specific chunk sizes.
    TODO: Only supports one dataset from the hdf5 file itself, labels etc need a custom implementation.
    """

    def __init__(self, data_dir='data', in_memory=False, force_download=False, force_preprocess=False):
        super().__init__(data_dir=data_dir, force_download=force_download, force_preprocess=force_preprocess)
        self._in_memory = in_memory

        # Load the entire dataset into memory if required
        if self._in_memory:
            # Only load the dataset once, no matter how many instances of the class are created.
            # data is stored on the underlying class at the _DATA property.
            if not hasattr(self.__class__, '_DATA'):
                print(f'[DATASET: {self.__class__.__name__}]: Loading...', end=' ')
                with h5py.File(self.dataset_path, 'r') as db:
                    self.__class__._DATA = np.array(db[self.hdf5_name])
                print('Loaded!')

    def __getitem__(self, idx):
        if self._in_memory:
            return self.__class__._DATA[idx]
        else:
            # open here for better multithreading support, saw this somewhere? check if correct.
            with h5py.File(self.dataset_path, 'r') as db:
                return db[self.hdf5_name][idx]

    def _preprocess_dataset(self, path_src, path_dst):
        import os
        from disent.dataset.util.hdf5 import hdf5_resave_dataset, hdf5_test_entries_per_second, bytes_to_human

        # resave datasets
        with h5py.File(path_src, 'r') as inp_data:
            with h5py.File(path_dst, 'w') as out_data:
                hdf5_resave_dataset(inp_data, out_data, self.hdf5_name, self.hdf5_chunk_size, self.hdf5_compression, self.hdf5_compression_lvl)
                # File Size:
                print(f'[FILE SIZES] IN: {bytes_to_human(os.path.getsize(path_src))} OUT: {bytes_to_human(os.path.getsize(path_dst))}\n')
                # Test Speed:
                print('[TESTING] Access Speed...', end=' ')
                print(f'Random Accesses Per Second: {hdf5_test_entries_per_second(out_data, self.hdf5_name, access_method="random"):.3f}')

    @property
    def hdf5_compression(self) -> 'str':
        return 'gzip'

    @property
    def hdf5_compression_lvl(self) -> int:
        return 4

    @property
    def hdf5_name(self) -> str:
        raise NotImplementedError()

    @property
    def hdf5_chunk_size(self) -> Tuple[int]:
        # dramatically affects access speed, but also compression ratio.
        raise NotImplementedError()


# ========================================================================= #
# paired factor of variation dataset                                        #
# ========================================================================= #


class PairedVariationDataset(Dataset):

    def __init__(self, dataset: Union[GroundTruthData, GroundTruthDataset], k=None, variation_factor_indices=None):
        """
        Dataset that pairs together samples with at most k differing factors of variation.

        dataset: A dataset that extends GroundTruthData
        k: An integer (k), None (k=d-1), or "uniform" (random k in range 1 to d-1)
        variation_factor_indices: The indices of the factors of variation that are samples between pairs, if None (all factors are sampled)
        """
        assert isinstance(dataset, GroundTruthDataset), 'passed object is not an instance of GroundTruthDataset'
        # wrapped dataset
        self._dataset: GroundTruthDataset = dataset
        # possible fixed dimensions between pairs
        self._variation_factor_indices = np.arange(self._dataset.data.num_factors) if (variation_factor_indices is None) else np.array(variation_factor_indices)
        # d
        self._num_variation_factors = len(self._variation_factor_indices)
        # number of varied factors between pairs
        self._k = self._num_variation_factors - 1 if (k is None) else k
        # verify k
        assert isinstance(k, str) or isinstance(k, int), f'k must be "uniform" or an integer 1 <= k <= d-1, d={self._num_variation_factors}'
        if isinstance(k, int):
            assert 1 <= k, 'k cannot be less than 1'
            assert k < self._num_variation_factors, f'all factors cannot be varied for each pair, k must be less than {self._num_variation_factors}'

    def __len__(self):
        # TODO: is dataset as big as the latent space OR as big as the orig.
        # return self._latent_space.size
        return self._dataset.data.size

    def __getitem__(self, idx):
        orig_factors, paired_factors = self.sample_pair_factors(idx)
        indices = self._dataset.data.pos_to_idx([orig_factors, paired_factors])
        return [self._dataset[idx] for idx in indices]

    def sample_pair_factors(self, idx):
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
        # get fixed or random k
        k = np.random.randint(1, self._dataset.data.num_factors) if self._k == 'uniform' else self._k
        # make k random indices not shared
        num_shared = self._num_variation_factors - k
        shared_indices = np.random.choice(self._variation_factor_indices, size=num_shared, replace=False)
        # resample paired item, differs by at most k factors of variation
        paired_factors = self._dataset.data.resampled_factors(orig_factors[np.newaxis, :], shared_indices)
        # return observations
        return orig_factors, paired_factors[0]


class RandomPairDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rand_idx = np.random.randint(len(self.dataset))
        return self.dataset[idx], self.dataset[rand_idx]



# ========================================================================= #
# END                                                                       #
# ========================================================================= #
