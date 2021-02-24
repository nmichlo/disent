"""
Flatness Metric
- Nathan Michlo et. al
"""

import logging
from dataclasses import dataclass
from typing import Iterable, Union, Dict

import torch
import numpy as np

from disent.dataset.groundtruth import GroundTruthDataset
from disent.util import chunked, to_numpy, colors

log = logging.getLogger(__name__)


# ========================================================================= #
# data storage                                                              #
# ========================================================================= #


@dataclass
class FactorsAggregates:
    # aggregates
    factors_ave_width: np.ndarray
    factors_ave_delta: np.ndarray
    factors_ave_deltas: np.ndarray
    # estimated version of factors_ave_width = factors_num_deltas * factors_ave_delta
    factors_ave_length: np.ndarray
    # metric
    factors_flatness: np.ndarray
    # extra
    factors_num_deltas: np.ndarray
# \/
@dataclass
class FactorAggregate:
    ave_width: torch.Tensor
    ave_delta: torch.Tensor
    ave_deltas: torch.Tensor
# \/
@dataclass
class FactorRepeats:
    deltas: torch.Tensor
    width: torch.Tensor


# ========================================================================= #
# flatness                                                                  #
# ========================================================================= #


def metric_flatness(
        ground_truth_dataset: GroundTruthDataset,
        representation_function: callable,
        factor_repeats: int = 1024,
        batch_size: int = 64,
):
    """
    Computes the flatness metric:
        approximately equal to: total_dim_width / (ave_point_dist_along_dim * num_points_along_dim)

    Complexity of this metric is:
        O(num_factors * ave_factor_size * repeats)
        eg. 9 factors * 64 indices on ave * 128 repeats = 73728 observations loaded from the dataset

    factor_repeats:
      - can go all the way down to about 64 and still get decent results.
      - 64 is accurate to about +- 0.01
      - 128 is accurate to about +- 0.003
      - 1024 is accurate to about +- 0.001

    Args:
      ground_truth_dataset: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and outputs a dim_representation sized representation for each observation.
      factor_repeats: how many times to repeat a traversal along each factors, these are then averaged together.
      batch_size: Batch size to process at any time while generating representations, should not effect metric results.
      p: how to calculate distances in the latent space, see torch.norm
    Returns:
      Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """

    # factors_mean_next_dists - an array for each factor, where each has the average distances to the next point for each factor index.
    # factors_ave_max_dist - a value for each factor, that is the average max width of the dimension
    # factors_ave_next_dist - a value for each factor, that is the average next dist of the dimension computed from factors_mean_next_dists
    factor_aggregates = aggregate_measure_distances_along_all_factors(
        ground_truth_dataset,
        representation_function,
        repeats=factor_repeats,
        batch_size=batch_size,
        ps=(1, 2),
    )

    results = {
        # metrics
        'flatness.ave_flatness_l1':  np.mean(factor_aggregates[1].factors_flatness),
        'flatness.ave_flatness_l2':  np.mean(factor_aggregates[2].factors_flatness),
        'flatness.ave_flatness':     np.mean(factor_aggregates[2].factors_ave_width / factor_aggregates[1].factors_ave_length),
        'flatness.ave_flatness_alt': np.mean(factor_aggregates[2].factors_ave_width) / np.mean(factor_aggregates[1].factors_ave_length),
        # sizes
        'flatness.ave_width_l1':  np.mean(factor_aggregates[1].factors_ave_width),
        'flatness.ave_width_l2':  np.mean(factor_aggregates[2].factors_ave_width),
        'flatness.ave_length_l1': np.mean(factor_aggregates[1].factors_ave_length),
        'flatness.ave_length_l2': np.mean(factor_aggregates[2].factors_ave_length),
    }

    return {k: float(v) for k, v in results.items()}


def aggregate_measure_distances_along_all_factors(
        ground_truth_dataset,
        representation_function,
        repeats: int,
        batch_size: int,
        ps: Iterable[Union[str, int]] = (1, 2),
) -> Dict[Union[str, int], FactorsAggregates]:
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # COMPUTE AGGREGATES FOR EACH FACTOR
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    all_aggregates = {p: FactorsAggregates([], [], [], None, None, None) for p in ps}
    # repeat for each factor
    for f_idx in range(ground_truth_dataset.num_factors):
        # repeatedly take measurements along a factor
        f_aggregates = aggregate_measure_distances_along_factor(
            ground_truth_dataset, representation_function,
            f_idx=f_idx, repeats=repeats, batch_size=batch_size, ps=ps,
        )
        # append all results
        for p, f_aggregate in f_aggregates.items():
            all_aggregates[p].factors_ave_deltas.append(f_aggregate.ave_deltas)
            all_aggregates[p].factors_ave_delta.append(f_aggregate.ave_delta)
            all_aggregates[p].factors_ave_width.append(f_aggregate.ave_width)
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # FINALIZE FOR EACH FACTOR
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    for p, aggregates in all_aggregates.items():
        aggregates: FactorsAggregates
        # convert everything to tensors
        aggregates.factors_ave_width = to_numpy(torch.stack(aggregates.factors_ave_width, dim=0))  # (f_dims,)
        aggregates.factors_ave_delta = to_numpy(torch.stack(aggregates.factors_ave_delta, dim=0))  # (f_dims,)
        aggregates.factors_ave_deltas = [to_numpy(deltas) for deltas in aggregates.factors_ave_deltas]  # each dimension has different sizes ((f_dims,) -> (f_size[i],))
        # check sizes
        assert aggregates.factors_ave_width.shape == (ground_truth_dataset.num_factors,)
        assert aggregates.factors_ave_delta.shape == (ground_truth_dataset.num_factors,)
        assert len(aggregates.factors_ave_deltas) == ground_truth_dataset.num_factors
        for i, factor_size in enumerate(ground_truth_dataset.factor_sizes):
            assert aggregates.factors_ave_deltas[i].shape == (factor_size,)
        # - - - - - - - - - - - - - - - - - #
        # COMPUTE THE ACTUAL METRICS!
        # get number of spaces, f_size[i] - 1
        aggregates.factors_num_deltas = np.array(ground_truth_dataset.factor_sizes) - 1
        assert np.all(aggregates.factors_num_deltas >= 1)
        assert aggregates.factors_num_deltas.shape == (ground_truth_dataset.num_factors,)
        # compute length
        # estimated version of factors_ave_width = factors_num_deltas * factors_ave_delta
        aggregates.factors_ave_length = aggregates.factors_num_deltas * aggregates.factors_ave_delta
        assert aggregates.factors_ave_length.shape == (ground_truth_dataset.num_factors,)
        # compute flatness ratio:
        # total_dim_width / total_dim_length
        aggregates.factors_flatness = aggregates.factors_ave_width / aggregates.factors_ave_length
        assert aggregates.factors_flatness.shape == (ground_truth_dataset.num_factors,)
        # - - - - - - - - - - - - - - - - - #
        # return values
    return all_aggregates


def aggregate_measure_distances_along_factor(
        ground_truth_dataset,
        representation_function,
        f_idx: int,
        repeats: int,
        batch_size: int,
        ps: Iterable[Union[str, int]] = (1, 2),
) -> Dict[Union[str, int], FactorAggregate]:
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # helper
    factor_size = ground_truth_dataset.factor_sizes[f_idx]
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # FEED FORWARD, COMPUTE ALL DELTAS & WIDTHS:
    # TODO: repeats don't make use of the full allowed batch size effectively
    #       if a factors dimensions are too small, then the remaining space is not used.
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # repeatedly take measurements along a factor, and return all the results
    # shapes: p -> (repeats, f_sizes[i]) & (repeats,)
    map_p_repeats = {p: FactorRepeats([], []) for p in ps}
    # repeat the calculations multiple times
    for _ in range(repeats):
        # generate repeated factors, varying one factor over a range
        sequential_zs = encode_all_along_factor(
            ground_truth_dataset,
            representation_function,
            f_idx=f_idx,
            batch_size=batch_size,
        )
        # for each distance measure compute everything
        for p in ps:
            # calculating the distances of their representations to the next values.
            deltas = measure_next_distances_along_encodings(sequential_zs, p=p)
            # calculate the distance between the furthest two points
            width = knn(x=sequential_zs, y=sequential_zs, k=1, largest=True, p=p).values.max()
            # append everyhing
            map_p_repeats[p].deltas.append(deltas)
            map_p_repeats[p].width.append(width)
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # AGGREGATE DATA
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    aggregates = {}
    for p, repeated in map_p_repeats.items():
        # convert everything to tensors
        repeated.deltas = torch.stack(repeated.deltas, dim=0)  # shape (repeats, f_sizes[i])
        repeated.width = torch.stack(repeated.width, dim=0)    # shape (repeats,)
        # check sizes
        assert repeated.deltas.shape == (repeats, factor_size)
        assert repeated.width.shape == (repeats,)
        # aggregate results
        ave_deltas = repeated.deltas.mean(dim=0)
        ave_width = repeated.width.mean(dim=0)
        assert ave_deltas.shape == (factor_size,)
        assert ave_width.shape == ()
        # we cant get the ave next dist directly because of cycles, so
        # we remove the max dist from those before calculating the mean
        smallest_deltas = repeated.deltas
        smallest_deltas = torch.topk(smallest_deltas, k=smallest_deltas.shape[-1]-1, dim=-1, largest=False, sorted=False).values
        assert smallest_deltas.shape == (repeats, factor_size - 1)
        # compute average of smallest deltas
        ave_delta = smallest_deltas.mean(dim=[0, 1])
        assert ave_delta.shape == ()
        # save the result
        aggregates[p] = FactorAggregate(
            ave_width=ave_width,    # (,)
            ave_delta=ave_delta,    # (,)
            ave_deltas=ave_deltas,  # (f_sizes[i],)
        )
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # we are done!
    return aggregates
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #


# ========================================================================= #
# ENCODE                                                                    #
# ========================================================================= #


def encode_all_along_factor(ground_truth_dataset, representation_function, f_idx: int, batch_size: int):
    f_size = ground_truth_dataset.factor_sizes[f_idx]
    # generate repeated factors, varying one factor over a range (f_size, f_dims)
    factors = range_along_repeated_factors(ground_truth_dataset, idx=f_idx, num=f_size)
    # get the representations of all the factors (f_size, z_size)
    sequential_zs = encode_all_factors(ground_truth_dataset, representation_function, factors=factors, batch_size=batch_size)
    return sequential_zs


def range_along_repeated_factors(ground_truth_dataset, idx: int, num: int) -> np.ndarray:
    """
    Aka. a traversal along a single factor
    """
    # make sequential factors, one randomly sampled list of
    # factors, then repeated, with one index mutated as if set by range()
    factors = ground_truth_dataset.sample_factors(size=1)
    factors = factors.repeat(num, axis=0)
    factors[:, idx] = np.arange(num)
    return factors


def encode_all_factors(ground_truth_dataset, representation_function, factors, batch_size: int) -> torch.Tensor:
    zs = []
    with torch.no_grad():
        for batch_factors in chunked(factors, chunk_size=batch_size):
            batch = ground_truth_dataset.dataset_batch_from_factors(batch_factors, mode='input')
            zs.append(representation_function(batch))
        zs = torch.cat(zs, dim=0)
    return zs


# ========================================================================= #
# DISTANCES                                                                 #
# ========================================================================= #


def measure_next_distances_along_encodings(sequential_zs, p='fro'):
    # find the distances to the next factors: z[i] - z[i+1]  (with wraparound)
    return torch.norm(sequential_zs - torch.roll(sequential_zs, -1, dims=0), dim=-1, p=p)


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


if __name__ == '__main__':
    import pytorch_lightning as pl
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from disent.data.groundtruth import XYObjectData, XYSquaresData
    from disent.dataset.groundtruth import GroundTruthDataset
    from disent.frameworks.vae.unsupervised import BetaVae
    from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder
    from disent.transform import ToStandardisedTensor

    def get_str(r):
        return ', '.join(f'{k}={v:6.4f}' for k, v in r.items())

    def print_r(name, steps, result, clr=colors.lYLW):
        print(f'{clr}{name:<13} ({steps:>04}): {get_str(result)}{colors.RST}')

    def calculate(name, steps, dataset, get_repr):
        r = metric_flatness(dataset, get_repr, factor_repeats=64)
        results.append((name, steps, r))
        print_r(name, steps, r, colors.lRED)
        print(colors.GRY, '='*100, colors.RST, sep='')
        return r

    class XYOverlapData(XYSquaresData):
        def __init__(self, square_size=8, grid_size=64, grid_spacing=None, num_squares=3, rgb=True):
            if grid_spacing is None:
                grid_spacing = (square_size+1) // 2
            super().__init__(square_size=square_size, grid_size=grid_size, grid_spacing=grid_spacing, num_squares=num_squares, rgb=rgb)

    results = []
    for data in [XYSquaresData(), XYOverlapData(), XYObjectData()]:
        dataset = GroundTruthDataset(data, transform=ToStandardisedTensor())
        dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, pin_memory=True)
        module = BetaVae(
            make_optimizer_fn=lambda params: Adam(params, lr=5e-4),
            make_model_fn=lambda: AutoEncoder(
                encoder=EncoderConv64(x_shape=data.x_shape, z_size=6, z_multiplier=2),
                decoder=DecoderConv64(x_shape=data.x_shape, z_size=6),
            ),
            cfg=BetaVae.cfg(beta=1)
        )
        # we cannot guarantee which device the representation is on
        get_repr = lambda x: module.encode(x.to(module.device))
        # PHASE 1, UNTRAINED
        calculate(data.__class__.__name__, 0, dataset, get_repr)
        # PHASE 2, LITTLE TRAINING
        pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, gpus=1, weights_summary=None).fit(module, dataloader)
        calculate(data.__class__.__name__, 256, dataset, get_repr)
        # PHASE 3, MORE TRAINING
        pl.Trainer(logger=False, checkpoint_callback=False, max_steps=2048, gpus=1, weights_summary=None).fit(module, dataloader)
        calculate(data.__class__.__name__, 256+2048, dataset, get_repr)
        results.append(None)

    for result in results:
        if result is None:
            print()
            continue
        (name, steps, result) = result
        print_r(name, steps, result, colors.lYLW)

