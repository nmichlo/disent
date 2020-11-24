from typing import List
import numpy as np


class BaseOptionEpisodesData(object):

    def __init__(self):
        self._episodes = self._load_episode_observations()
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
        # get ordered list of random indices
        indices = set()
        while len(indices) < n:
            indices.add(np.random.randint(0, len(episode)))
        # sort indices from highest to lowest.
        # - anchor is the newest
        # - positive is close in the past
        # - negative is far in the past
        indices = sorted(indices)[::-1]
        print(indices)
        # return indices
        return tuple([episode[i] for i in indices])

    def _load_episode_observations(self) -> List[np.ndarray]:
        raise NotImplementedError
