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

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from disent.data.dataset import Hdf5Dataset
import matplotlib.pyplot as plt

from disent.data.groundtruth import GroundTruthData
from disent.data.groundtruth import Shapes3dData


class TransformDataset(GroundTruthData):

    # TODO: all data should be datasets
    # TODO: file preparation should be separate from datasets
    # TODO: disent/data should be datasets, and disent/datasets should be samplers that wrap disent/data

    def __init__(self, base_data: GroundTruthData, transform=None):
        self.base_data = base_data
        self._transform = transform
        super().__init__()

    @property
    def factor_names(self) -> Tuple[str, ...]:
        return self.base_data.factor_names

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        return self.base_data.factor_sizes

    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self.base_data.observation_shape

    def __getitem__(self, idx):
        item = self.base_data[idx]
        if self._transform is not None:
            item = self._transform(item)
        return item


class AdversarialOptimizedData(TransformDataset):

    def __init__(self, h5_path: str, base_data: GroundTruthData, transform=None):
        # normalize hd5f data
        def _normalize_hdf5(x):
            c, h, w = x.shape
            if c in (1, 3):
                return np.moveaxis(x, 0, -1)
            return x
        # get the data
        self.hdf5_data = Hdf5Dataset(h5_path, transform=_normalize_hdf5)
        # initialize
        super().__init__(base_data=base_data, transform=transform)
        # checks
        assert len(self.base_data) == len(self.hdf5_data)

    def __getitem__(self, idx):
        item = self.hdf5_data[idx]
        if self._transform is not None:
            item = self._transform(item)
        return item


if __name__ == '__main__':

    def ave_pairwise_dist(data, n_samples=1000):
        # get stats
        diff = []
        for i in range(n_samples):
            a, b = np.random.randint(0, len(data), size=2)
            a, b = data[a], data[b]
            diff.append(F.mse_loss(a, b, reduction='mean').item())
        return np.mean(diff)

    def plot_samples(data, name=None):
        # get image
        img = torchvision.utils.make_grid([data[i*1000] for i in range(9)], nrow=3)
        img = torch.moveaxis(img, 0, -1).numpy()
        # plot
        if name is not None:
            plt.title(name)
        plt.imshow(img)
        plt.show()


    def main():
        base_data = TransformDataset(Shapes3dData(in_memory=False, prepare=True), transform=torchvision.transforms.ToTensor())
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
