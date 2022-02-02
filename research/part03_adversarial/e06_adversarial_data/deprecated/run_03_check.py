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


"""
Check the adversarial data generated in previous exerpiments
- This is old and outdated...
- Should use `plot02_data_distances/run_plot_traversal_dists.py` instead!
"""


import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt

from disent.dataset.data import Shapes3dData
from research.util._data import AdversarialOptimizedData


if __name__ == '__main__':

    def ave_pairwise_dist(data, n_samples=1000):
        """
        Get the average distance between observations in the dataset
        """
        # get stats
        diff = []
        for i in range(n_samples):
            a, b = np.random.randint(0, len(data), size=2)
            a, b = data[a], data[b]
            diff.append(F.mse_loss(a, b, reduction='mean').item())
        return np.mean(diff)

    def plot_samples(data, name=None):
        """
        Display random observations from the dataset
        """
        # get image
        img = torchvision.utils.make_grid([data[i*1000] for i in range(9)], nrow=3)
        img = torch.moveaxis(img, 0, -1).numpy()
        # plot
        if name is not None:
            plt.title(name)
        plt.imshow(img)
        plt.show()


    def main():
        base_data = Shapes3dData(in_memory=False, prepare=True, transform=torchvision.transforms.ToTensor())
        plot_samples(base_data)
        print(ave_pairwise_dist(base_data))

        for path in [
            'out/overlap/fixed_masked_const_shapes3d_adam_0.01_True_None_mse_12288_shuffle_5120_10240_None_0.1_False_8_125.hdf5',
            'out/overlap/fixed_masked_randm_shapes3d_adam_0.01_True_None_mse_12288_shuffle_5120_10240_None_None_False_8_125.hdf5',
            'out/overlap/noise_unmask_randm_shapes3d_adam_0.01_False_0.001_mse_12288_shuffle_5120_10240_None_None_False_8_125.hdf5',
            'out/overlap/noise_unmask_randm_shapes3d_adam_0.01_False_0.1_mse_12288_shuffle_5120_10240_None_None_False_8_125.hdf5',
        ]:
            data = AdversarialOptimizedData(path, base_data, transform=torchvision.transforms.ToTensor())
            plot_samples(data)
            print(ave_pairwise_dist(data))

    main()
