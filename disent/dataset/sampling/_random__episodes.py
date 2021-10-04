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

from disent.dataset.data import BaseEpisodesData
from disent.dataset.sampling._base import BaseDisentSampler
from disent.util.math.random import sample_radius as sample_radius_fn


# ========================================================================= #
# Episode Sampler                                                           #
# ========================================================================= #


class RandomEpisodeSampler(BaseDisentSampler):

    def uninit_copy(self) -> 'RandomEpisodeSampler':
        return RandomEpisodeSampler(
            num_samples=self.num_samples,
            sample_radius=self._sample_radius,
        )

    def __init__(self, num_samples=1, sample_radius=None):
        super().__init__(num_samples=num_samples)
        self._sample_radius = sample_radius
        # checks
        if sample_radius is not None:
            if sample_radius >= 0:
                if num_samples > sample_radius:
                    raise RuntimeError(f'sample radius: {sample_radius} cannot be less than the number of requested samples: {num_samples}')

    def _init(self, dataset):
        assert isinstance(dataset, BaseEpisodesData), f'data ({type(dataset)}) is not an instance of {BaseEpisodesData}'
        # TODO: reference to dataset is not ideal here
        self._dataset = dataset

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _sample_idx(self, idx):
        # TODO: are we actually sampling distances correctly?
        # sample for observations
        episode, idx, offset = self._dataset.get_episode_and_idx(idx)
        indices = self.sample_episode_indices(episode, idx, n=self._num_samples, radius=self._sample_radius)
        # transform back to original indices
        return tuple(i + offset for i in indices)

    def sample_episode_indices(self, episode, idx, n=1, radius=None):
        # TODO: update this to use the same API
        #       as ground truth triplet and pair.
        # default value
        if radius is None:
            radius = len(episode)
        elif radius < 0:
            # TODO: I think this is the same as None? radius > len(episode) makes no difference?
            radius = len(episode) + radius + 1
        assert n <= len(episode)
        assert n <= radius
        # sample values
        attempts, indices = 0, {idx}
        while (len(indices) < n) and (attempts < n * 100):
            indices.add(sample_radius_fn(idx, low=0, high=len(episode), r_low=0, r_high=radius))
            attempts += 1
        # checks
        if len(indices) != n:
            raise RuntimeError('consider increasing the radius')
        # sort indices from highest to lowest.
        # - anchor is the newest
        # - positive is close in the past
        # - negative is far in the past
        return sorted(indices)[::-1]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
