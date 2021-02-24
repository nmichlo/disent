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

from typing import List, Tuple
import numpy as np

from disent.dataset.groundtruth._triplet import sample_radius
from disent.util import LengthIter


class BaseOptionEpisodesData(LengthIter):

    def __init__(self):
        self._episodes = self._load_episode_observations()
        assert len(self._episodes) > 0, 'There must be at least one episode!'
        # total length
        self._lengths = np.array([len(episode) for episode in self._episodes])
        self._length = np.sum(self._lengths)
        self._weights = self._lengths / self._length

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # this can be slow!
        episode, idx, _ = self.get_episode_and_idx(idx)
        return episode[idx]

    def get_episode_and_idx(self, idx) -> Tuple[np.ndarray, int, int]:
        assert idx >= 0, 'Negative indices are not supported.'
        # linear search for episode & shift idx accordingly
        # TODO: This could be better...
        # TODO: add caching?
        offset = 0
        for episode in self._episodes:
            length = len(episode)
            if idx < length:
                break
            else:
                offset += length
                idx -= length
        # return found
        return episode, idx, offset

    @staticmethod
    def sample_episode_indices(episode, idx, n=1, radius=None):
        # TODO: update this to use the same API
        #       as ground truth triplet and pair.
        # default value
        if radius is None:
            radius = len(episode)
        elif radius < 0:
            radius = len(episode) + radius + 1
        assert n <= len(episode)
        assert n <= radius
        # sample values
        indices = {idx}
        while len(indices) < n:
            indices.add(sample_radius(idx, low=0, high=len(episode), r_low=0, r_high=radius))
        # sort indices from highest to lowest.
        # - anchor is the newest
        # - positive is close in the past
        # - negative is far in the past
        return sorted(indices)[::-1]

    def _load_episode_observations(self) -> List[np.ndarray]:
        raise NotImplementedError
