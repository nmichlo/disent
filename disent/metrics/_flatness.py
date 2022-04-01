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
Flatness Metric
- Michlo et al.
  https://github.com/nmichlo/msc-research
"""

import logging
import math
from typing import Iterable
from typing import Tuple
from typing import Union

import torch

from disent.metrics.utils import make_metric
from disent.util.deprecate import deprecated
from torch.utils.data.dataloader import default_collate

from disent.dataset import DisentDataset
from disent.util.iters import iter_chunks


log = logging.getLogger(__name__)


# ========================================================================= #
# flatness                                                                  #
# ========================================================================= #


@make_metric('flatness', fast_kwargs=dict(repeats=128))
def metric_flatness(
        dataset: DisentDataset,
        representation_function: callable,
        repeats: int = 1024,
        batch_size: int = 64,
):
    """
    Michlo et al.
    https://github.com/nmichlo/msc-research

    Computes the flatness metric:
        approximately equal to: total_dim_width / (ave_point_dist_along_dim * num_points_along_dim)

    Complexity of this metric is:
        O(num_factors * ave_factor_size * repeats)
        eg. 9 factors * 64 indices on ave * 128 repeats = 73728 observations loaded from the dataset

    repeats:
      - can go all the way down to about 64 and still get decent results.
      - 64 is accurate to about +- 0.01
      - 128 is accurate to about +- 0.003
      - 1024 is accurate to about +- 0.001

    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observations as input and outputs a dim_representation sized representation for each observation.
      repeats: how many times to repeat a traversal along each factors, these are then averaged together.
      batch_size: Batch size to process at any time while generating representations, should not effect metric results.
      p: how to calculate distances in the latent space, see torch.norm
    Returns:
      Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    p_fs_measures = aggregate_measure_distances_along_all_factors(dataset, representation_function, repeats=repeats, batch_size=batch_size, ps=(1, 2))
    # get info
    factor_sizes = dataset.gt_data.factor_sizes
    # aggregate data
    results = {
        # main metric -- also measures axis-alignment
        'flatness.ave_flatness':    compute_flatness(widths=p_fs_measures[2]['fs_ave_widths'], lengths=p_fs_measures[1]['fs_ave_lengths'], factor_sizes=factor_sizes),
        # optional metrics
        'flatness.ave_flatness_l1': compute_flatness(widths=p_fs_measures[1]['fs_ave_widths'], lengths=p_fs_measures[1]['fs_ave_lengths'], factor_sizes=factor_sizes),
        'flatness.ave_flatness_l2': compute_flatness(widths=p_fs_measures[2]['fs_ave_widths'], lengths=p_fs_measures[2]['fs_ave_lengths'], factor_sizes=factor_sizes),
        # distances
        'flatness.ave_width_l1':    torch.mean(filter_inactive_factors(p_fs_measures[1]['fs_ave_widths'], factor_sizes=factor_sizes)),
        'flatness.ave_width_l2':    torch.mean(filter_inactive_factors(p_fs_measures[2]['fs_ave_widths'], factor_sizes=factor_sizes)),
        'flatness.ave_length_l1':   torch.mean(filter_inactive_factors(p_fs_measures[1]['fs_ave_lengths'], factor_sizes=factor_sizes)),
        'flatness.ave_length_l2':   torch.mean(filter_inactive_factors(p_fs_measures[2]['fs_ave_lengths'], factor_sizes=factor_sizes)),
    }
    # convert values from torch
    return {k: float(v) for k, v in results.items()}


def compute_flatness(widths, lengths, factor_sizes):
    widths = filter_inactive_factors(widths, factor_sizes)
    lengths = filter_inactive_factors(lengths, factor_sizes)
    # checks
    assert torch.all(widths >= 0)
    assert torch.all(lengths >= 0)
    assert torch.all(torch.eq(widths == 0, lengths == 0))
    # update scores
    widths[lengths == 0] = 0
    lengths[lengths == 0] = 1
    # compute flatness
    return (widths / lengths).mean()


def filter_inactive_factors(tensor, factor_sizes):
    factor_sizes = torch.tensor(factor_sizes, device=tensor.device)
    assert torch.all(factor_sizes >= 1)
    # remove
    active_factors = torch.nonzero(factor_sizes-1, as_tuple=True)
    return tensor[active_factors]


def aggregate_measure_distances_along_all_factors(
        dataset: DisentDataset,
        representation_function,
        repeats: int,
        batch_size: int,
        ps: Iterable[Union[str, int]] = (1, 2),
) -> dict:
    # COMPUTE AGGREGATES FOR EACH FACTOR
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    fs_p_measures = [
        aggregate_measure_distances_along_factor(dataset, representation_function, f_idx=f_idx, repeats=repeats, batch_size=batch_size, ps=ps)
        for f_idx in range(dataset.gt_data.num_factors)
    ]

    # FINALIZE FOR EACH FACTOR
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    p_fs_measures = {}
    for p, fs_measures in default_collate(fs_p_measures).items():
        fs_ave_widths = fs_measures['ave_width']
        # get number of spaces deltas (number of points minus 1)
        # compute length: estimated version of factors_ave_width = factors_num_deltas * factors_ave_delta
        _fs_num_deltas = torch.as_tensor(dataset.gt_data.factor_sizes, device=fs_ave_widths.device) - 1
        _fs_ave_deltas = fs_measures['ave_delta']
        fs_ave_lengths = _fs_num_deltas * _fs_ave_deltas
        # update
        p_fs_measures[p] = {'fs_ave_widths': fs_ave_widths, 'fs_ave_lengths': fs_ave_lengths}
    return p_fs_measures


def aggregate_measure_distances_along_factor(
        dataset: DisentDataset,
        representation_function,
        f_idx: int,
        repeats: int,
        batch_size: int,
        ps: Iterable[Union[str, int]] = (1, 2),
        cycle_fail: bool = False,
) -> dict:
    f_size = dataset.gt_data.factor_sizes[f_idx]

    if f_size == 1:
        if cycle_fail:
            raise ValueError(f'dataset factor size is too small for flatness metric with cycle_normalize enabled! size={f_size} < 2')
        zero = torch.as_tensor(0., device=get_device(dataset, representation_function))
        return {p: {'ave_width': zero.clone(), 'ave_delta': zero.clone()} for p in ps}

    # FEED FORWARD, COMPUTE ALL DELTAS & WIDTHS - For each distance measure
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    p_measures: list = [{} for _ in range(repeats)]
    for measures in p_measures:
        # generate repeated factors, varying one factor over the entire range
        zs_traversal = encode_all_along_factor(dataset, representation_function, f_idx=f_idx, batch_size=batch_size)
        # for each distance measure compute everything
        # - width: calculate the distance between the furthest two points
        # - deltas: calculating the distances of their representations to the next values.
        # - cycle_normalize: we cant get the ave next dist directly because of cycles, so we remove the largest dist
        for p in ps:
            deltas_next = torch.norm(torch.roll(zs_traversal, -1, dims=0) - zs_traversal, dim=-1, p=p)  # next | shape: (factor_size, z_size)
            deltas_prev = torch.norm(torch.roll(zs_traversal,  1, dims=0) - zs_traversal, dim=-1, p=p)  # prev | shape: (factor_size, z_size)
            # values needed for flatness
            width  = knn(x=zs_traversal, y=zs_traversal, k=1, largest=True, p=p).values.max()           # shape: (,)
            min_deltas = torch.topk(deltas_next, k=f_size-1, dim=-1, largest=False, sorted=False)       # shape: (factor_size-1, z_size)
            # TODO: other measures can be added:
            #       1. multivariate skewness
            #       2. normality measure
            #       3. independence
            #       4. menger curvature (Cayley-Menger Determinant?)
            # save variables
            measures[p] = {'widths': width, 'deltas': min_deltas.values}

    # AGGREGATE DATA - For each distance measure
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    return {
        p: {
            'ave_width': measures['widths'].mean(dim=0),       # shape: (repeats,) -> ()
            'ave_delta': measures['deltas'].mean(dim=[0, 1]),  # shape: (repeats, factor_size - 1) -> ()
        } for p, measures in default_collate(p_measures).items()
    }


# ========================================================================= #
# ENCODE                                                                    #
# ========================================================================= #


def encode_all_along_factor(dataset: DisentDataset, representation_function, f_idx: int, batch_size: int, return_batch: bool = False):
    # generate repeated factors, varying one factor over a range (f_size, f_dims)
    factors = dataset.gt_data.sample_random_factor_traversal(f_idx=f_idx)
    # get the representations of all the factors (f_size, z_size)
    # * if return_batch is False: return sequential_zs
    # * if return_batch is True: return (sequential_zs, sequential_batch)
    return encode_all_factors(dataset, representation_function, factors=factors, batch_size=batch_size, return_batch=return_batch)


def encode_all_factors(dataset: DisentDataset, representation_function, factors, batch_size: int, return_batch: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    zs = []
    xs = []
    with torch.no_grad():
        for batch_factors in iter_chunks(factors, chunk_size=batch_size):
            batch = dataset.dataset_batch_from_factors(batch_factors, mode='input')
            z = representation_function(batch)
            zs.append(z)
            if return_batch:
                xs.append(batch)
    # handle case
    if return_batch:
        # might be on different devices
        return torch.cat(zs, dim=0), torch.cat(xs, dim=0)
    else:
        return torch.cat(zs, dim=0)


def get_device(dataset: DisentDataset, representation_function):
    # this is a hack...
    return representation_function(dataset.dataset_sample_batch(1, mode='input')).device


# ========================================================================= #
# DISTANCES                                                                 #
# ========================================================================= #


def knn(x, y, k: int = None, largest=False, p='fro'):
    assert 0 < k <= y.shape[0]
    # check input vectors, must be array of vectors
    assert 2 == x.ndim == y.ndim
    assert x.shape[1:] == y.shape[1:]
    # compute distances between each and every pair
    dist_mat = x[:, None, ...] - y[None, :, ...]
    dist_mat = torch.norm(dist_mat, dim=-1, p=p)
    # return closest distances
    return torch.topk(dist_mat, k=k, dim=-1, largest=largest, sorted=True)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


# if __name__ == '__main__':
#     import pytorch_lightning as pl
#     from torch.optim import Adam
#     from torch.utils.data import DataLoader
#     from disent.data.groundtruth import XYObjectData, XYSquaresData
#     from disent.dataset.groundtruth import GroundTruthDataset, GroundTruthDatasetPairs
#     from disent.frameworks.vae import BetaVae
#     from disent.frameworks.vae import AdaVae
#     from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder
#     from disent.transform import ToImgTensorF32
#     from disent.util import colors
#     from disent.util import Timer
#
#     def get_str(r):
#         return ', '.join(f'{k}={v:6.4f}' for k, v in r.items())
#
#     def print_r(name, steps, result, clr=colors.lYLW, t: Timer = None):
#         print(f'{clr}{name:<13} ({steps:>04}){f" {colors.GRY}[{t.pretty}]{clr}" if t else ""}: {get_str(result)}{colors.RST}')
#
#     def calculate(name, steps, dataset, get_repr):
#         global aggregate_measure_distances_along_factor
#         with Timer() as t:
#             r = metric_flatness(dataset, get_repr, repeats=64, batch_size=64)
#         results.append((name, steps, r))
#         print_r(name, steps, r, colors.lRED, t=t)
#         print(colors.GRY, '='*100, colors.RST, sep='')
#         return r
#
#     class XYOverlapData(XYSquaresData):
#         def __init__(self, square_size=8, image_size=64, grid_spacing=None, num_squares=3, rgb=True):
#             if grid_spacing is None:
#                 grid_spacing = (square_size+1) // 2
#             super().__init__(square_size=square_size, image_size=image_size, grid_spacing=grid_spacing, num_squares=num_squares, rgb=rgb)
#
#     # datasets = [XYObjectData(rgb=False, palette='white'), XYSquaresData(), XYOverlapData(), XYObjectData()]
#     datasets = [XYObjectData()]
#
#     results = []
#     for data in datasets:
#         dataset = GroundTruthDatasetPairs(data, transform=ToImgTensorF32())
#         dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, pin_memory=True)
#         module = AdaVae(
#             model=AutoEncoder(
#                 encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
#                 decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
#             ),
#             cfg=AdaVae.cfg(beta=0.001, loss_reduction='mean', optimizer=torch.optim.Adam, optimizer_kwargs=dict(lr=5e-4))
#         )
#         # we cannot guarantee which device the representation is on
#         get_repr = lambda x: module.encode(x.to(module.device))
#         # PHASE 1, UNTRAINED
#         pl.Trainer(logger=False, checkpoint_callback=False, fast_dev_run=True, gpus=1, weights_summary=None).fit(module, dataloader)
#         module = module.to('cuda')
#         calculate(data.__class__.__name__, 0, dataset, get_repr)
#         # PHASE 2, LITTLE TRAINING
#         pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, gpus=1, weights_summary=None).fit(module, dataloader)
#         calculate(data.__class__.__name__, 256, dataset, get_repr)
#         # PHASE 3, MORE TRAINING
#         pl.Trainer(logger=False, checkpoint_callback=False, max_steps=2048, gpus=1, weights_summary=None).fit(module, dataloader)
#         calculate(data.__class__.__name__, 256+2048, dataset, get_repr)
#         results.append(None)
#
#     for result in results:
#         if result is None:
#             print()
#             continue
#         (name, steps, result) = result
#         print_r(name, steps, result, colors.lYLW)
