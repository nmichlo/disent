import logging
import os
from abc import ABCMeta
from typing import List, Tuple
import h5py

from disent.data.util.in_out import basename_from_url, download_file, ensure_dir_exists
from disent.data.util.state_space import StateSpace


log = logging.getLogger(__name__)


# ========================================================================= #
# ground truth data                                                         #
# ========================================================================= #


class GroundTruthData(StateSpace):

    def __init__(self):
        assert len(self.factor_names) == len(self.factor_sizes), 'Dimensionality mismatch of FACTOR_NAMES and FACTOR_DIMS'
        super().__init__(factor_sizes=self.factor_sizes)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Overrides                                                             #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def factor_names(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        raise NotImplementedError()

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        # shape as would be for a non-batched observation
        # eg. H x W x C
        raise NotImplementedError()

    @property
    def x_shape(self) -> Tuple[int, ...]:
        # shape as would be for a single observation in a torch batch
        # eg. C x H x W
        shape = self.observation_shape
        return shape[-1], *shape[:-1]

    def __getitem__(self, idx):
        raise NotImplementedError


# ========================================================================= #
# dataset helpers                                                           #
# ========================================================================= #


class DownloadableGroundTruthData(GroundTruthData, metaclass=ABCMeta):

    def __init__(self, data_dir='data/dataset', force_download=False):
        super().__init__()
        # paths
        self._data_dir = ensure_dir_exists(data_dir)
        self._data_paths = [os.path.join(self._data_dir, basename_from_url(url)) for url in self.dataset_urls]
        # meta
        self._force_download = force_download
        # DOWNLOAD
        self._do_download_dataset()

    def _do_download_dataset(self):
        for path, url in zip(self.dataset_paths, self.dataset_urls):
            no_data = not os.path.exists(path)
            # download data
            if self._force_download or no_data:
                download_file(url, path)

    @property
    def dataset_paths(self) -> List[str]:
        """path that the data should be loaded from in the child class"""
        return self._data_paths

    @property
    def dataset_urls(self) -> List[str]:
        raise NotImplementedError()


class PreprocessedDownloadableGroundTruthData(DownloadableGroundTruthData, metaclass=ABCMeta):

    def __init__(self, data_dir='data/dataset', force_download=False, force_preprocess=False):
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
    def _data_path(self):
        assert len(self.dataset_paths) == 1
        return self.dataset_paths[0]

    @property
    def dataset_urls(self):
        return [self.dataset_url]

    @property
    def dataset_url(self):
        raise NotImplementedError()

    @property
    def dataset_path(self):
        """path that the dataset should be loaded from in the child class"""
        return self._proc_path

    @property
    def dataset_path_unprocessed(self):
        return self._data_path

    def _preprocess_dataset(self, path_src, path_dst):
        raise NotImplementedError()


class Hdf5PreprocessedGroundTruthData(PreprocessedDownloadableGroundTruthData, metaclass=ABCMeta):
    """
    Automatically download and pre-process an hdf5 dataset into the specific chunk sizes.
    TODO: Only supports one dataset from the hdf5 file itself, labels etc need a custom implementation.
    """

    def __init__(self, data_dir='data/dataset', in_memory=False, force_download=False, force_preprocess=False):
        super().__init__(data_dir=data_dir, force_download=force_download, force_preprocess=force_preprocess)
        self._in_memory = in_memory

        # Load the entire dataset into memory if required
        if self._in_memory:
            # Only load the dataset once, no matter how many instances of the class are created.
            # data is stored on the underlying class at the _DATA property.
            # TODO: this is weird
            if not hasattr(self.__class__, '_DATA'):
                log.info(f'[DATASET: {self.__class__.__name__}]: Loading...')
                # Often the (non-chunked) dataset will be optimized for random accesses,
                # while the unprocessed (chunked) dataset will be better for sequential reads.
                with h5py.File(self.dataset_path, 'r') as db:
                    # indexing dataset objects returns numpy array
                    # instantiating np.array from the dataset requires double memory.
                    self.__class__._DATA = db[self.hdf5_name][:]
                log.info(f'[DATASET: {self.__class__.__name__}]: Loaded!')

    def __getitem__(self, idx):
        if self._in_memory:
            return self.__class__._DATA[idx]
        else:
            # This actually doesnt seem too slow
            with h5py.File(self.dataset_path, 'r', libver='latest', swmr=True) as f:
                return f[self.hdf5_name][idx]

    def _preprocess_dataset(self, path_src, path_dst):
        import os
        from disent.data.util.hdf5 import hdf5_resave_dataset, hdf5_test_entries_per_second, bytes_to_human

        # resave datasets
        with h5py.File(path_src, 'r') as inp_data:
            with h5py.File(path_dst, 'w') as out_data:
                hdf5_resave_dataset(inp_data, out_data, self.hdf5_name, self.hdf5_chunk_size, self.hdf5_compression, self.hdf5_compression_lvl)
                # File Size:
                log.info(f'[FILE SIZES] IN: {bytes_to_human(os.path.getsize(path_src))} OUT: {bytes_to_human(os.path.getsize(path_dst))}\n')
                # Test Speed:
                log.info('[TESTING] Access Speed...')
                log.info(f'Random Accesses Per Second: {hdf5_test_entries_per_second(out_data, self.hdf5_name, access_method="random"):.3f}')

    @property
    def hdf5_compression(self) -> 'str':
        return 'gzip'

    @property
    def hdf5_compression_lvl(self) -> int:
        # default is 4, max of 9 doesnt seem to add much cpu usage on read, but its not worth it data wise?
        return 4

    @property
    def hdf5_name(self) -> str:
        raise NotImplementedError()

    @property
    def hdf5_chunk_size(self) -> Tuple[int]:
        # dramatically affects access speed, but also compression ratio.
        raise NotImplementedError()


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
