from typing import List, Tuple
import numpy as np


class OptionEpisodesPickledData:

    def __init__(self, episodes_pickle_file: str):
        self._episodes = self._load_episode_observations_from_options(episodes_pickle_file)
        # total length
        self._lengths = np.array([len(episode) for episode in self._episodes])
        self._length = np.sum(self._lengths)
        self._weights = self._lengths / self._length

    @property
    def episodes(self):
        return self._episodes[:]

    def get_random_episode(self, weighted=True) -> np.ndarray:
        if weighted:
            return np.random.choice(self._episodes, p=self._weights)
        else:
            return np.random.choice(self._episodes)

    def get_random_observation(self, weighted=True, n=1):
        episode = self.get_random_episode(weighted=weighted)
        # choose observations
        assert len(episode) >= n
        # get list of random indices
        indices = set()
        while len(indices) < n:
            indices.add(np.random.randint(0, len(episode)))
        indices = sorted(indices)
        # return indices
        return tuple([episode[i] for i in indices])

    @staticmethod
    def _load_episode_observations_from_options(pickle_file_path='temp/monte.pkl') -> List[np.ndarray]:
        import pickle
        # load the raw data!
        with open(pickle_file_path, 'rb') as f:
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