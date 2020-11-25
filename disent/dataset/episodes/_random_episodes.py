from torch.utils.data import IterableDataset
from disent.data.episodes import BaseOptionEpisodesData
# from disent.util import TempNumpySeed


class RandomEpisodeDataset(IterableDataset):

    def __init__(self, episodes_data: BaseOptionEpisodesData, num_samples=1, weight_episodes=True):
        assert isinstance(episodes_data, BaseOptionEpisodesData), f'episodes_data ({type(episodes_data)}) is not an instance of {BaseOptionEpisodesData}'
        self._episodes = episodes_data
        self._num_samples = num_samples
        self._weight_episodes = weight_episodes

    def _next(self):
        return self._episodes.get_random_observation(
            weighted=self._weight_episodes,
            n=self._num_samples
        )

    def __iter__(self):
        while True:
            yield self._next()

    # def __getitem__(self, idx):
    #     with TempNumpySeed(seed=idx):
    #         return self._next()

