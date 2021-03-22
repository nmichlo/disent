#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

from disent.data.episodes import BaseOptionEpisodesData
from disent.dataset.random import RandomDataset


class RandomEpisodeDataset(RandomDataset):

    # type hint, override RandomDataset
    _data: BaseOptionEpisodesData

    def __init__(
            self,
            data: BaseOptionEpisodesData,
            transform=None,
            augment=None,
            num_samples=1,
            sample_radius=None
    ):
        super().__init__(
            data=data,
            transform=transform,
            augment=augment,
            num_samples=num_samples
        )
        # checks
        assert isinstance(self._data, BaseOptionEpisodesData), f'data ({type(self._data)}) is not an instance of {BaseOptionEpisodesData}'
        self._sample_radius = sample_radius

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def __getitem__(self, idx):
        # sample for observations
        # TODO: are we actually sampling distances correctly?
        episode, idx, offset = self._data.get_episode_and_idx(idx)
        indices = self._data.sample_episode_indices(episode, idx, n=self._num_samples, radius=self._sample_radius)
        # transform back to original indices
        indices = [i + offset for i in indices]
        # TODO: this is inefficient, we have to perform multiple searches for the same thing!
        return self.dataset_get_observation(*indices)
