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

import numpy as np
from torch.utils.data import Dataset
from disent.data.episodes import BaseOptionEpisodesData
from disent.dataset._augment_util import AugmentableDataset
from disent.util import LengthIter


class RandomEpisodeDataset(Dataset, LengthIter, AugmentableDataset):

    def __init__(
            self,
            episodes_data: BaseOptionEpisodesData,
            transform=None,
            augment=None,
            num_samples=1,
            sample_radius=None
    ):
        assert isinstance(episodes_data, BaseOptionEpisodesData), f'episodes_data ({type(episodes_data)}) is not an instance of {BaseOptionEpisodesData}'
        self._episodes = episodes_data
        self._num_samples = num_samples
        self._sample_radius = sample_radius
        # augmentable dataset
        self._transform = transform
        self._augment = augment

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Augmentable Dataset Overrides                                         #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def transform(self):
        return self._transform

    @property
    def augment(self):
        return self._augment

    def _get_augmentable_observation(self, idx):
        return self._episodes[idx]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def __len__(self):
        return len(self._episodes)

    def __getitem__(self, idx):
        # sample for observations
        episode, idx, offset = self._episodes.get_episode_and_idx(idx)
        indices = self._episodes.sample_episode_indices(episode, idx, n=self._num_samples, radius=self._sample_radius)
        # transform back to original indices
        indices = [i + offset for i in indices]
        # TODO: this is inefficient, we have to perform multiple searches for the same thing!
        return self.dataset_get_observation(*indices)


