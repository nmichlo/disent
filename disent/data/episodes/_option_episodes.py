import os
from typing import List, Tuple
import numpy as np
from disent.data.episodes._base import BaseOptionEpisodesData
from disent.data.util.in_out import download_file, basename_from_url
import logging

log = logging.getLogger(__file__)


class OptionEpisodesPickledData(BaseOptionEpisodesData):

    def __init__(self, required_file: str):
        assert os.path.isabs(required_file), f'{required_file=} must be an absolute path.'
        self._required_file = required_file
        # load data
        super().__init__()

    def _load_episode_observations(self) -> List[np.ndarray]:
        import pickle
        # load the raw data!
        with open(self._required_file, 'rb') as f:
            # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
            # - Each element in the root list represents an episode
            # - An episode is a list containing many executed options
            # - Each option is a tuple containing:
            #     1. The option name
            #     2. The option id
            #     3. A list of ground truth states covering the option execution. Each ground truth state is a dictionary
            #     4. A list of environment images covering the option execution
            # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
            # Episode = List[Options]
            #   Options = List[Option]
            #     Option  = Tuple[OptionName, OptionId, GroundTruthStates, ObservedStates]
            #       OptionName        = str
            #       OptionId          = int
            #       GroundTruthStates = List[dict]
            #       ObservedStates    = List[np.ndarray]
            # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
            raw_episodes = pickle.load(f)
        # check variables
        option_ids_to_names = {}
        ground_truth_keys = None
        observation_shape = None
        # load data
        episodes = []
        for i, raw_episode in enumerate(raw_episodes):
            rollout = []
            for j, raw_option in enumerate(raw_episode):
                # GET: option info
                raw_option: Tuple[str, int, List[dict], List[np.ndarray]]
                option_name, option_id, ground_truth_states, observed_states = raw_option
                # CHECK: number of observations
                assert len(ground_truth_states) == len(observed_states)
                # CHECK: option ids and names
                if option_id not in option_ids_to_names:
                    option_ids_to_names[option_id] = option_name
                else:
                    assert option_ids_to_names[option_id] == option_name
                # CHECK: ground truth keys
                if ground_truth_keys is None:
                    ground_truth_keys = set(ground_truth_states[0].keys())
                else:
                    for gt_state in ground_truth_states:
                        assert ground_truth_keys == gt_state.keys()
                # CHECK: observation shapes
                if observation_shape is None:
                    observation_shape = observed_states[0].shape
                else:
                    for observation in observed_states:
                        assert observation.shape == observation_shape
                # APPEND: all observations into one long episode
                rollout.extend(observed_states)
                # cleanup unused memory! This is not ideal, but works well.
                raw_episode[j] = None
            # make the long episode!
            episodes.append(np.array(rollout))
            # cleanup unused memory! This is not ideal, but works well.
            raw_episodes[i] = None
        # done!
        return episodes


class OptionEpisodesDownloadZippedPickledData(OptionEpisodesPickledData):

    def __init__(self, required_file: str, download_url=None, force_download=False):
        self._download_and_extract_if_needed(download_url=download_url, required_file=required_file, force_download=force_download)
        super().__init__(required_file=required_file)

    def _download_and_extract_if_needed(self, download_url: str, required_file: str, force_download: bool):
        # TODO: this function should probably be moved to the io file.
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
        # skip if no download url
        if not isinstance(download_url, str):
            return
        # download file, but skip if file already exists
        save_path = os.path.join(os.path.dirname(required_file), basename_from_url(download_url))
        if force_download or not os.path.exists(save_path):
            log.info(f'Downloading: {download_url=} to {save_path=}')
            download_file(download_url, save_path=save_path)
            log.info(f'Downloaded!')
        # check that the downloaded file exists
        assert os.path.exists(save_path), 'The file specified for download does not exist!'
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
        # unzip data
        if (save_path != required_file) and not os.path.exists(required_file):
            if save_path.endswith('.tar.xz'):
                import tarfile
                log.info(f'Extracting: {save_path=} to {required_file=}')
                with tarfile.open(save_path) as f:
                    f.extractall(os.path.dirname(required_file))
                log.info(f'Extracted!')
            else:
                raise IOError(f'Unsupported extension for: {save_path}')
        # check that everything exists
        assert os.path.exists(required_file), 'The required file does not exist after downloading and extracting if necessary!'
        # ~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~


