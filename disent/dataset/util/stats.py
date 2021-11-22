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

import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from disent.util.function import wrapped_partial


# ========================================================================= #
# COMPUTE DATASET STATS                                                     #
# ========================================================================= #


@torch.no_grad()
def compute_data_mean_std(
    data,
    batch_size: int = 256,
    num_workers: int = min(os.cpu_count(), 16),
    progress: bool = False,
    chn_is_last: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Input data when collected using a DataLoader should return
    `torch.Tensor`s, output mean and std are an `np.ndarray`s
    """
    loader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    if progress:
        from tqdm import tqdm
        loader = tqdm(loader, desc=f'{data.__class__.__name__} stats', total=(len(data) + batch_size - 1) // batch_size)
    # reduction dims
    dims = (1, 2) if chn_is_last else (2, 3)
    # collect obs means & stds
    img_means, img_stds = [], []
    for batch in loader:
        assert isinstance(batch, torch.Tensor), f'batch must be an instance of torch.Tensor, got: {type(batch)}'
        assert batch.ndim == 4, f'batch shape must be: (B, C, H, W), got: {tuple(batch.shape)}'
        batch = batch.to(torch.float64)
        img_means.append(torch.mean(batch, dim=dims))
        img_stds.append(torch.std(batch, dim=dims))
    # aggregate obs means & stds
    mean = torch.mean(torch.cat(img_means, dim=0), dim=0)
    std  = torch.mean(torch.cat(img_stds, dim=0), dim=0)
    # checks!
    assert mean.ndim == 1
    assert std.ndim == 1
    # done!
    return mean.numpy(), std.numpy()


# ========================================================================= #
# HELPER                                                                    #
# ========================================================================= #


if __name__ == '__main__':

    def main(progress=False):
        from disent.dataset import data
        from disent.dataset.transform import ToImgTensorF32

        for data_cls in [
            # groundtruth -- impl
            data.Cars3dData,
            data.DSpritesData,
            data.SmallNorbData,
            data.Shapes3dData,
            # groundtruth -- impl synthetic
            data.XYObjectData,
            data.XYObjectShadedData,
            # large datasets
            (data.Mpi3dData, dict(subset='toy',       in_memory=True)),
            (data.Mpi3dData, dict(subset='realistic', in_memory=True)),
            (data.Mpi3dData, dict(subset='real',      in_memory=True)),
        ]:
            from disent.dataset.transform import ToImgTensorF32
            # get arguments
            if isinstance(data_cls, tuple):
                data_cls, kwargs = data_cls
            else:
                data_cls, kwargs = data_cls, {}
            # Most common standardized way of computing the mean and std over observations
            # resized to 64px in size of dtype float32 in the range [0, 1].
            data = data_cls(transform=ToImgTensorF32(size=64), **kwargs)
            mean, std = compute_data_mean_std(data, progress=progress)
            # results!
            print(f'{data.__class__.__name__} - {data.name} - {kwargs}:\n    mean: {mean.tolist()}\n    std: {std.tolist()}')

    # RUN!
    main()


# ========================================================================= #
# RESULTS: 2021-11-12                                                       #
# ========================================================================= #


# Cars3dData - cars3d - {}:
#     mean: [0.8976676149976628, 0.8891658020067508, 0.885147515814868]
#     std: [0.22503195531503034, 0.2399461278981261, 0.24792106319684404]
# DSpritesData - dsprites - {}:
#     mean: [0.042494423521889584]
#     std: [0.19516645880626055]
# SmallNorbData - smallnorb - {}:
#     mean: [0.7520918401088603]
#     std: [0.09563879016827263]
# Shapes3dData - 3dshapes - {}:
#     mean: [0.502584966788819, 0.5787597566089667, 0.6034499731859578]
#     std: [0.2940814043555559, 0.34439790875172144, 0.3661685981524748]

# XYBlocksData - xyblocks - {}:
#     mean: [0.10040509259259259, 0.10040509259259259, 0.10040509259259259]
#     std: [0.21689087652106678, 0.21689087652106676, 0.21689087652106678]
# XYObjectData - xy_object - {}:
#     mean: [0.009818761549013288, 0.009818761549013288, 0.009818761549013288]
#     std: [0.052632363725245844, 0.05263236372524584, 0.05263236372524585]
# XYObjectShadedData - xy_object - {}:
#     mean: [0.009818761549013288, 0.009818761549013288, 0.009818761549013288]
#     std: [0.052632363725245844, 0.05263236372524584, 0.05263236372524585]
# XYSquaresData - xy_squares - {}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920855, 0.12403473458920854, 0.12403473458920854]
# XYSquaresMinimalData - xy_squares_minimal - {}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920855, 0.12403473458920854, 0.12403473458920854]
# XColumnsData - x_columns - {}:
#     mean: [0.125, 0.125, 0.125]
#     std: [0.33075929223788925, 0.3307592922378891, 0.3307592922378892]

# XYSquaresData - xy_squares - {'grid_size': 8, 'grid_spacing': 8}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920855, 0.12403473458920854, 0.12403473458920854]
# overlap between squares for reconstruction loss, 7 < 8
# XYSquaresData - xy_squares - {'grid_size': 8, 'grid_spacing': 7}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920854, 0.12403473458920854, 0.12403473458920854]
# overlap between squares for reconstruction loss, 6 < 8
# XYSquaresData - xy_squares - {'grid_size': 8, 'grid_spacing': 6}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920854, 0.12403473458920854, 0.12403473458920855]
# overlap between squares for reconstruction loss, 5 < 8
# XYSquaresData - xy_squares - {'grid_size': 8, 'grid_spacing': 5}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920855, 0.12403473458920855, 0.12403473458920854]
# overlap between squares for reconstruction loss, 4 < 8
# XYSquaresData - xy_squares - {'grid_size': 8, 'grid_spacing': 4}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920855, 0.12403473458920854, 0.12403473458920854]
# overlap between squares for reconstruction loss, 3 < 8
# XYSquaresData - xy_squares - {'grid_size': 8, 'grid_spacing': 3}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920854, 0.12403473458920854, 0.12403473458920854]
# overlap between squares for reconstruction loss, 2 < 8
# XYSquaresData - xy_squares - {'grid_size': 8, 'grid_spacing': 2}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920854, 0.12403473458920854, 0.12403473458920854]
# overlap between squares for reconstruction loss, 1 < 8
# XYSquaresData - xy_squares - {'grid_size': 8, 'grid_spacing': 1}:
#     mean: [0.015625, 0.015625, 0.015625]
#     std: [0.12403473458920855, 0.12403473458920855, 0.12403473458920855]
# XYSquaresData - xy_squares - {'rgb': False}:
#     mean: [0.046146392822265625]
#     std: [0.2096506119375896]

# Mpi3dData - mpi3d_toy - {'subset': 'toy', 'in_memory': True}:
#     mean: [0.22681593831231503, 0.22353985202496676, 0.22666059934624702]
#     std: [0.07854112062669572, 0.07319301658077378, 0.0790763900050426]
# Mpi3dData - mpi3d_realistic - {'subset': 'realistic', 'in_memory': True}:
#     mean: [0.18240164396358813, 0.20723063241107917, 0.1820551008003256]
#     std: [0.09511163559287175, 0.10128881101801782, 0.09428244469525177]
# Mpi3dData - mpi3d_real - {'subset': 'real', 'in_memory': True}:
#     mean: [0.13111154099374112, 0.16746449372488892, 0.14051725201807627]
#     std: [0.10137409845578041, 0.10087824338375781, 0.10534121043187629]


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
