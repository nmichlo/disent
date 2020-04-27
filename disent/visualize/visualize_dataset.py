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

from disent.dataset import make_ground_truth_dataset
from disent.visualize import visualize_util
import numpy as np
from six.moves import range

from disent.dataset.util.io import ensure_dir_exists


# ========================================================================= #
# Visualise Ground Truth Datasets                                           #
# ========================================================================= #


def visualize_dataset(dataset_name, output_path, num_animations=5, num_frames=20, fps=10):
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
    data = make_ground_truth_dataset(dataset_name, try_in_memory=False)

    # Create output folder if necessary.
    path = ensure_dir_exists(output_path, dataset_name)

    # Create still images.
    for i in range(data.num_factors):
        factors = data.sample_factors(16)
        indices = [j for j in range(data.num_factors) if i != j]
        factors[:, indices] = factors[0, indices]
        images = data.sample_observations_from_factors(factors)
        visualize_util.grid_save_images(images, os.path.join(path, "variations_of_factor%s.png" % i))

    # Create animations.
    for i in range(num_animations):
        base_factor = data.sample_factors(1)
        images = []
        for j, factor_size in enumerate(data.factor_sizes):
            factors = np.repeat(base_factor, num_frames, axis=0)
            factors[:, j] = visualize_util.cycle_factor(base_factor[0, j], factor_size, num_frames)
            images.append(data.sample_observations_from_factors(factors))

        visualize_util.save_animation(
            np.array(images),
            os.path.join(path, "animation%d.gif" % i),
            fps=fps
        )


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    visualize_dataset('dsprites', 'data/output')