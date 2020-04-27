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

"""Methods to visualize latent factors in the data sets."""
import os
from typing import Union

from disent.dataset import make_ground_truth_data
from disent.dataset.ground_truth.base import GroundTruthData
from disent.loss.loss import anneal_step, lerp
from disent.visualize import visualize_util
import numpy as np
from six.moves import range

from disent.dataset.util.io import ensure_dir_exists


# ========================================================================= #
# Visualise Ground Truth Datasets                                           #
# ========================================================================= #

def _get_data(data: Union[str, GroundTruthData]) -> GroundTruthData:
    if isinstance(data, str):
        data = make_ground_truth_data(data, try_in_memory=False)
    return data


def visualise_get_still_images(data: Union[str, GroundTruthData], num_samples=16, mode='lerp'):
    data = _get_data(data)
    # Create still images per factor of variation
    factor_images = []
    for i, size in enumerate(data.factor_sizes):
        factors = data.sample_factors(num_samples)
        # only allow the current index to vary, copy the first to all others
        indices = [j for j in range(data.num_factors) if i != j]
        factors[:, indices] = factors[0, indices]

        if mode == 'sample':
            pass
        elif mode == 'sample_ordered':
            # like sample, but ordered
            factors[:, i] = sorted(factors[:, i])
        elif mode == 'lerp':
            # like spread but much better
            factors[:, i] = [round(anneal_step(0, size-1, j, num_samples-1)) for j in range(num_samples)]
        elif mode == 'spread':
            # 1, 3, 5, 7, 9 (like sequential below, but use larger step size if size bigger than samples)
            indices = np.tile(np.arange(size), (num_samples + size - 1) // size)
            factors[:, i] = np.sort(indices[::max(1, size//num_samples)][:num_samples])
        elif mode == 'sequential':
            # 1, 2, 3, 4, 5 (repeat if not size is not as big as number of samples)
            indices = np.tile(np.arange(size), (num_samples + size - 1) // size)
            factors[:, i] = np.sort(indices[:num_samples])
        else:
            raise KeyError(f'Unsupported mode: {mode}')

        # sample new observations
        images = data.sample_observations_from_factors(factors)
        factor_images.append(images)
    # return all
    return np.array(factor_images)

def visualise_get_animations(data: Union[str, GroundTruthData], num_animations=5, num_frames=20):
    data = _get_data(data)
    # Create animations.
    animations = []
    for i in range(num_animations):
        base_factor = data.sample_factors(1)
        images = []
        for j, factor_size in enumerate(data.factor_sizes):
            factors = np.repeat(base_factor, num_frames, axis=0)
            factors[:, j] = visualize_util.cycle_factor(base_factor[0, j], factor_size, num_frames)
            images.append(data.sample_observations_from_factors(factors))
        animations.append(images)
    # return all
    return np.array(animations)


def visualize_dataset(data, output_path=None, num_animations=5, num_frames=20, fps=10, mode='lerp'):
    """Visualizes the data set by saving images to output_path.

    For each latent factor, outputs 16 images where only that latent factor is
    varied while all others are kept constant.

    Args:
      dataset_name: String with name of dataset as defined in named_data.py.
      output_path: String with path in which to create the visualizations.
      num_animations: Integer with number of distinct animations to create.
      num_frames: Integer with number of frames in each animation.
      fps: Integer with frame rate for the animation.
    """
    data = _get_data(data)

    # Create output folder if necessary.
    path = ensure_dir_exists(output_path, data.__class__.__name__[:-4].lower())

    print(f'[VISUALISE] saving to: {path}')

    # Save still images.
    for i, images in enumerate(visualise_get_still_images(data, num_samples=16, mode=mode)):
        visualize_util.grid_save_images(images, os.path.join(path, f"variations_of_factor_{i}.png"))

    # Save animations.
    for i, images in enumerate(visualise_get_animations(data, num_animations=num_animations, num_frames=num_frames)):
        visualize_util.save_animation(
            np.array(images),
            os.path.join(path, f"animation_{i}.gif"),
            fps=fps
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    visualise_get_still_images('xygrid', num_samples=5, mode='sample')
    visualise_get_still_images('xygrid', num_samples=5, mode='sample_ordered')
    visualise_get_still_images('xygrid', num_samples=5, mode='sequential')
    visualise_get_still_images('xygrid', num_samples=5, mode='spread')
    visualise_get_still_images('xygrid', num_samples=5, mode='lerp')

    # visualize_dataset('3dshapes', 'data/output')