# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions that are useful for the different metrics."""

import numpy as np
from tqdm import tqdm

from disent.dataset.groundtruth import GroundTruthDataset
from disent.util import to_numpy


# ========================================================================= #
# utils                                                                   #
# ========================================================================= #


def generate_batch_factor_code(
        ground_truth_dataset: GroundTruthDataset,
        representation_function,
        num_points,
        batch_size,
        show_progress=False,
):
    """Sample a single training sample based on a mini-batch of ground-truth data.
    Args:
      ground_truth_dataset: GroundTruthData to be sampled from.
      representation_function: Function that takes observation as input and outputs a representation.
      num_points: Number of points to sample.
      batch_size: Batchsize to sample points.
      show_progress: if a progress bar should be shown
    Returns:
      representations: Codes (num_codes, num_points)-np array.
      factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    representations = None
    factors = None
    i = 0
    with tqdm(total=num_points, disable=not show_progress) as bar:
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            current_observations, current_factors = ground_truth_dataset.dataset_sample_batch_with_factors(num_points_iter, mode='input')
            if i == 0:
                factors = current_factors
                representations = to_numpy(representation_function(current_observations.cuda()))
            else:
                factors = np.vstack((factors, current_factors))
                representations = np.vstack((representations, to_numpy(representation_function(current_observations.cuda()))))
            i += num_points_iter
            bar.update(num_points_iter)
    return np.transpose(representations), np.transpose(factors)


def obtain_representation(observations, representation_function, batch_size):
    """"Obtain representations from observations.
    Args:
      observations: Observations for which we compute the representation.
      representation_function: Function that takes observation as input and
        outputs a representation.
      batch_size: Batch size to compute the representation.
    Returns:
      representations: Codes (num_codes, num_points)-Numpy array.
    """
    representations = None
    # TODO: use chunked
    num_points = observations.shape[0]
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_observations = observations[i:i + num_points_iter]
        if i == 0:
            representations = to_numpy(representation_function(current_observations))
        else:
            representations = np.vstack((representations, to_numpy(representation_function(current_observations))))
        i += num_points_iter
    return np.transpose(representations)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
