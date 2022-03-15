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

import logging
import os

import imageio
import numpy as np
import torch
from matplotlib import pyplot as plt

import disent.registry as R
import research.code.util as H
from disent.util.visualize.vis_img import torch_to_images
from research.code import register_to_disent


# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    # matplotlib style
    plt.style.use(os.path.join(os.path.dirname(__file__), '../../code/util/gadfly.mplstyle'))

    def main():
        register_to_disent()

        # ['box_r31', 'gau_r31', 'xy8_r47', 'xy1_r47']
        for key in R.KERNELS.examples:
            kernel: torch.Tensor = R.KERNELS[key]
            # normalize kernel
            b, c, h, w = kernel.shape
            assert b == 1
            # convert kernel to image
            image = torch_to_images(kernel[0], in_min=kernel.min(), in_max=kernel.max(), always_rgb=True, to_numpy=True)
            if 'box_' in key:
                image = np.full_like(image, fill_value=128)
            # save and show
            path = H.make_rel_path_add_ext('plots', f'kernel_{key}', ext='.png')
            print(f'saving: {path}')
            imageio.imsave(path, image)
            H.plt_imshow(image, show=True)

    main()


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
