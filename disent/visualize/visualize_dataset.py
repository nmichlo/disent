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
#
# ========================================================================
#
# ADAPTED FROM Google's disentanglement_lib:
# https://github.com/google-research/disentanglement_lib
#
# Modified for pytorch and disent by Nathan Michlo


"""Methods to visualize latent factors in the data sets."""

import os
import logging
from typing import Union

from disent.dataset.ground_truth_data.base_data import GroundTruthData
from disent.frameworks.unsupervised.betavae import lerp_step
from disent.util import TempNumpySeed, to_numpy
from disent.visualize import visualize_util
import numpy as np

from disent.dataset.util.in_out import ensure_dir_exists
from disent.dataset import DEPRICATED_as_data

log = logging.getLogger(__name__)

# ========================================================================= #
# Visualise Ground Truth Datasets                                           #
# ========================================================================= #


def sample_dataset_still_images(data: Union[str, GroundTruthData], num_samples=16, mode='spread', seed=None):
    data = DEPRICATED_as_data(data)
    # Create still images per factor of variation
    factor_images = []
    for i, size in enumerate(data.factor_sizes):
        with TempNumpySeed(seed, offset=i):
            factors = data.sample_factors(num_samples)
        # only allow the current index to vary, copy the first to all others
        indices = [j for j in range(data.num_factors) if i != j]
        factors[:, indices] = factors[0, indices]
        # get values for current factor of variation
        if mode == 'sample_unordered':
            pass
        elif mode == 'sample':
            factors[:, i] = sorted(factors[:, i])
        elif mode == 'spread':
            # spread all available factors over all samples with linear interpolation.
            # if num_samples == factor_size then values == 0, 1, 2, 3, 4, 5, ...
            factors[:, i] = [round(lerp_step(0, size - 1, j, num_samples - 1)) for j in range(num_samples)]
        else:
            raise KeyError(f'Unsupported mode: {repr(mode)} not in {{\'sample\', \'sample_unordered\', \'spread\'}}')
        # get and store observations
        images = data.sample_observations_from_factors(factors)
        factor_images.append(images)
    # return all
    return to_numpy(factor_images)


def sample_dataset_animations(data: Union[str, GroundTruthData], num_animations=5, num_frames=20, seed=None):
    data = DEPRICATED_as_data(data)
    # Create animations.
    animations = []
    for animation_num in range(num_animations):
        with TempNumpySeed(seed, offset=animation_num):
            base_factor = data.sample_factors(1)
        images = []
        for i, factor_size in enumerate(data.factor_sizes):
            factors = np.repeat(base_factor, num_frames, axis=0)
            factors[:, i] = visualize_util.cycle_factor(base_factor[0, i], factor_size, num_frames)
            images.append(data.sample_observations_from_factors(factors))
        animations.append(images)
    # return all
    return to_numpy(animations)


def save_dataset_visualisations(data: Union[str, GroundTruthData], output_path=None, num_animations=5, num_frames=20, fps=10, mode='spread'):
    """Visualizes the data set by saving images to output_path.

    For each latent factor, outputs 16 images where only that latent factor is
    varied while all others are kept constant.

    Args:
      data: String with name of data as defined in named_data.py.
      output_path: String with path in which to create the visualizations.
      num_animations: Integer with number of distinct animations to create.
      num_frames: Integer with number of frames in each animation.
      fps: Integer with frame rate for the animation.
      mode: still image mode, see function visualise_get_still_images
    """
    data = DEPRICATED_as_data(data)
    # Create output folder if necessary.
    path = ensure_dir_exists(output_path, data.__class__.__name__[:-4].lower())
    log.info(f'Saving visualisations to: {path}')
    # Save still images.
    for i, images in enumerate(sample_dataset_still_images(data, num_samples=16, mode=mode)):
        visualize_util.minimal_square_save_images(images, os.path.join(path, f"variations_of_factor_{i}.png"))
    # Save animations.
    for i, images in enumerate(sample_dataset_animations(data, num_animations=num_animations, num_frames=num_frames)):
        visualize_util.save_gridified_animation(images, os.path.join(path, f"animation_{i}.gif"), fps=fps)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    save_dataset_visualisations('3dshapes', 'data/output')