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
import random
from typing import List
from typing import Literal
from typing import Union

import numpy as np
import pytest

from disent.dataset import DisentDataset
from disent.dataset.data import BaseEpisodesData
from disent.dataset.sampling import *
from disent.dataset.data import XYObjectData


class TestEpisodesData(BaseEpisodesData):
    def _load_episode_observations(self) -> List[np.ndarray]:
        return [
            np.random.randn(57, 3, 64, 64),
            np.random.randn(42, 3, 64, 64),
            np.random.randn(26, 3, 64, 64),
            np.random.randn(97, 3, 64, 64),
        ]


@pytest.mark.parametrize(['dataset', 'num_samples', 'check_mode', 'sampler'], [
    [XYObjectData(), 1, 'first', SingleSampler()],
    [XYObjectData(), 1, 'first', GroundTruthSingleSampler()],

    # original AdaVAE sampling method
    [XYObjectData(), 2, 'first', GroundTruthPairOrigSampler(p_k=1)],
    [XYObjectData(), 2, 'first', GroundTruthPairOrigSampler(p_k=2)],
    [XYObjectData(), 2, 'first', GroundTruthPairOrigSampler(p_k=-1)],

    # TODO: consider removing the pair sampler... it is difficult to maintain and confusing
    [XYObjectData(), 2, 'first', GroundTruthPairSampler()],

    # TODO: consider removing the triplet sampler... it is difficult to maintain and confusing
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler()],
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler(swap_metric='k')],
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler(swap_metric='manhattan')],
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler(swap_metric='manhattan_norm')],
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler(swap_metric='euclidean')],
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler(swap_metric='euclidean_norm')],
    # [XYObjectData(), 3, 'first', GroundTruthTripleSampler(n_k_sample_mode='offset')],  # TODO: these are broken
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler(n_k_sample_mode='bounded_below')],
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler(n_k_sample_mode='random')],
    # [XYObjectData(), 3, 'first', GroundTruthTripleSampler(n_radius_sample_mode='offset')],  # TODO: these are broken
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler(n_radius_sample_mode='bounded_below')],
    [XYObjectData(), 3, 'first', GroundTruthTripleSampler(n_radius_sample_mode='random')],

    [XYObjectData(), 1, 'first', GroundTruthDistSampler(num_samples=1)],
    [XYObjectData(), 2, 'first', GroundTruthDistSampler(num_samples=2)],
    [XYObjectData(), 3, 'first', GroundTruthDistSampler(num_samples=3)],
    [XYObjectData(), 3, 'first', GroundTruthDistSampler(num_samples=3, triplet_sample_mode='random')],
    [XYObjectData(), 3, 'first', GroundTruthDistSampler(num_samples=3, triplet_sample_mode='factors')],
    [XYObjectData(), 3, 'first', GroundTruthDistSampler(num_samples=3, triplet_sample_mode='manhattan')],
    [XYObjectData(), 3, 'first', GroundTruthDistSampler(num_samples=3, triplet_sample_mode='manhattan_scaled')],
    [XYObjectData(), 3, 'first', GroundTruthDistSampler(num_samples=3, triplet_sample_mode='combined')],
    [XYObjectData(), 3, 'first', GroundTruthDistSampler(num_samples=3, triplet_sample_mode='combined_scaled')],

    [XYObjectData(), 1, 'first', RandomSampler(num_samples=1)],
    [XYObjectData(), 2, 'first', RandomSampler(num_samples=2)],
    [XYObjectData(), 3, 'first', RandomSampler(num_samples=3)],

    [TestEpisodesData(), 1, 'any', RandomEpisodeSampler(num_samples=1)],
    [TestEpisodesData(), 2, 'any', RandomEpisodeSampler(num_samples=2)],
    [TestEpisodesData(), 3, 'any', RandomEpisodeSampler(num_samples=3)],
    [TestEpisodesData(), 3, 'any', RandomEpisodeSampler(num_samples=3, sample_radius=3)],  # sample_radius >= num_samples
    [TestEpisodesData(), 3, 'any', RandomEpisodeSampler(num_samples=3, sample_radius=4)],
    [TestEpisodesData(), 3, 'any', RandomEpisodeSampler(num_samples=3, sample_radius=-1)],
])
def test_samplers(dataset, num_samples: int, check_mode: Union[Literal['first'], Literal['any']], sampler: BaseDisentSampler):
    # check dataset
    wrapper = DisentDataset(dataset, sampler)
    assert len(wrapper) == len(dataset)
    assert sampler.num_samples == num_samples
    # check dataset init & samples
    for batch in wrapper:
        assert isinstance(batch, dict)
        assert len(batch['x_targ']) == sampler.num_samples
        break
    # check sample
    def check_samples(i: int):
        indices = sampler(i)
        assert isinstance(indices, tuple)
        assert len(indices) == num_samples
        if check_mode == 'first':
            assert i == indices[0]
        elif check_mode == 'any':
            assert i in indices
        else:  # pragma: no cover
            raise RuntimeError('test mode is invalid!')
    # check indices
    check_samples(0)
    check_samples(len(dataset) - 1)
    for i in range(10):
        check_samples(random.randint(0, len(dataset)-1))
