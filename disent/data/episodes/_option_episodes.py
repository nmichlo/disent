from typing import List, Tuple
import numpy as np
from disent.data.episodes._base import BaseOptionEpisodesData


class OptionEpisodesPickledData(BaseOptionEpisodesData):

    def __init__(self, episodes_pickle_file: str = 'temp/monte.pkl'):
        self._episodes_pickle_file = episodes_pickle_file
        # load data
        super().__init__()

    def _load_episode_observations(self) -> List[np.ndarray]:
        import pickle
        # load the raw data!
        with open(self._episodes_pickle_file, 'rb') as f:
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
