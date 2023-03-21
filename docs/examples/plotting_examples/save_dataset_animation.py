#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2022 Nathan Juraj Michlo
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

import os

import imageio
import numpy as np

import docs.examples.plotting_examples.util as H
from disent.util.inout.paths import ensure_parent_dir_exists
from disent.util.seeds import TempNumpySeed

if __name__ == "__main__":
    OUTPUT_DIR = os.path.abspath(os.path.join(__file__, "..", "plots/animations"))
    FRAMES_PER_TRAVERSAL = 18
    FRAMES_PER_SECOND = 8

    for data_name, base_factors in [
        ("xysquares_8x8", [5, 5, 1, 5, 5, 1]),  # ('x_R', 'y_R', 'x_G', 'y_G', 'x_B', 'y_B')
        ("dsprites", [2, 3, 4, 16, 20]),  # ('shape', 'scale', 'orientation', 'position_x', 'position_y')
        ("cars3d", [2, 2, 79]),  # ('elevation', 'azimuth', 'object_type')
        ("shapes3d", [1, 6, 9, 4, 3, 2]),  # ('floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation')
        ("smallnorb", [2, 4, 1, 2, 3]),  # ('category', 'instance', 'elevation', 'rotation', 'lighting')
        (
            "mpi3d_toy",
            [2, 3, 1, 2, 1, 11, 10],
        ),  # ('object_color', 'object_shape', 'object_size', 'camera_height', 'background_color', 'first_dof', 'second_dof') (4, 4, 2, 3, 3, 40, 40)
        (
            "mpi3d_realistic",
            [2, 3, 1, 2, 1, 11, 10],
        ),  # ('object_color', 'object_shape', 'object_size', 'camera_height', 'background_color', 'first_dof', 'second_dof') (4, 4, 2, 3, 3, 40, 40)
        (
            "mpi3d_real",
            [2, 3, 1, 2, 1, 11, 10],
        ),  # ('object_color', 'object_shape', 'object_size', 'camera_height', 'background_color', 'first_dof', 'second_dof') (4, 4, 2, 3, 3, 40, 40)
        ("sprites", 777),
    ]:
        data = H.make_data(data_name, transform_mode="none")

        # get starting point
        if isinstance(base_factors, int):
            with TempNumpySeed(base_factors):
                base_factors = data.sample_factors()
        base_factors = np.array(base_factors)
        print(f"{data_name}:", base_factors.tolist(), data.factor_names, data.factor_sizes)

        # generate and append the factor traversals
        frames = []
        for f_idx in range(data.num_factors):
            traversal = data.sample_random_factor_traversal(
                f_idx=f_idx,
                base_factors=base_factors,
                start_index=base_factors[f_idx],
                num=FRAMES_PER_TRAVERSAL,
                mode="cycle_from_start",
            )
            frames.extend(data[i] for i in data.pos_to_idx(traversal))

        # save the animation
        path = ensure_parent_dir_exists(OUTPUT_DIR, f"animation__{data_name}.gif")
        imageio.mimsave(path, frames, fps=FRAMES_PER_SECOND)
        print(f"saved: {path}")
