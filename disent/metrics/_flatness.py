"""
Flatness Metric
- Nathan Michlo et. al
"""

import logging
from pprint import pprint

import torch
import numpy as np

from disent.dataset.groundtruth import GroundTruthDataset
from disent.util import chunked, to_numpy, colors

log = logging.getLogger(__name__)


# ========================================================================= #
# dci                                                                       #
# ========================================================================= #


def metric_flatness(
        ground_truth_dataset: GroundTruthDataset,
        representation_function: callable,
        factor_repeats: int = 1024,  # can go all the way down to about 64 and still get decent results. 128 is accurate to about ~2 decimal places.
        batch_size: int = 64,
        p='fro',
        return_extra=False,
):
    """
    Computes the flatness metric:
    approximately equal to: total_dim_width / (ave_point_dist_along_dim * num_points_along_dim)

    Complexity of this metric is:
    O(num_factors * ave_factor_size * repeats)
    eg. 9 factors * 64 indices on ave * 128 repeats = 73728 observations loaded from the dataset

    Args:
      ground_truth_dataset: GroundTruthData to be sampled from.
      representation_function: Function that takes observations as input and outputs a dim_representation sized representation for each observation.
      factor_repeats: how many times to repeat a traversal along each factors, these are then averaged together.
      batch_size: Batch size to process at any time while generating representations, should not effect metric results.
      p: how to calculate distances in the latent space, see torch.norm
      show_progress: if a progress bar should be shown while computing the metric
    Returns:
      Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """

    # factors_mean_next_dists - an array for each factor, where each has the average distances to the next point for each factor index.
    # factors_ave_max_dist - a value for each factor, that is the average max width of the dimension
    # factors_ave_next_dist - a value for each factor, that is the average next dist of the dimension computed from factors_mean_next_dists
    factors_mean_next_dists, factors_ave_max_dist, factors_ave_next_dist = aggregate_measure_distances_along_all_factors(
        ground_truth_dataset, representation_function,
        repeats=factor_repeats, batch_size=batch_size, p=p,
    )

    # compute flatness ratio:
    # total_dim_width / (ave_point_dist_along_dim * num_points_along_dim)
    factors_flatness = factors_ave_max_dist / (factors_ave_next_dist * torch.tensor(ground_truth_dataset.factor_sizes))

    return {
        'flatness': float(factors_flatness.mean()),
        'flatness.median': float(factors_flatness.median()),
        'flatness.max': float(factors_flatness.max()),
        'flatness.min': float(factors_flatness.min()),
        **({} if (not return_extra) else {
            'flatness.factors.flatness': to_numpy(factors_flatness),
            'flatness.factors.next_dists': [to_numpy(dists) for dists in factors_mean_next_dists],
            'flatness.factors.ave_max_dist': to_numpy(factors_ave_max_dist),
            'flatness.factors.ave_next_dist': to_numpy(factors_ave_max_dist),
        }),
    }


def aggregate_measure_distances_along_all_factors(
        ground_truth_dataset, representation_function,
        repeats: int, batch_size: int, p='fro',
):
    factors_mean_next_dists = []
    factors_ave_max_dist = []
    factors_ave_next_dist = []
    # append all
    for f_idx in range(ground_truth_dataset.num_factors):
        # repeatedly take measurements along a factor
        mean_next_dists, ave_max_dist, ave_next_dist = aggregate_measure_distances_along_factor(
            ground_truth_dataset, representation_function,
            f_idx=f_idx, repeats=repeats, batch_size=batch_size, p=p,
        )
        # append all results
        factors_mean_next_dists.append(mean_next_dists)
        factors_ave_max_dist.append(ave_max_dist)
        factors_ave_next_dist.append(ave_next_dist)
    # combine everything
    # we ignore "factors_mean_next_dists" because each dimension has different sizes (f_dims, f_size[i])
    factors_ave_max_dist = torch.stack(factors_ave_max_dist, dim=0)  # (f_dims,)
    factors_ave_next_dist = torch.stack(factors_ave_next_dist, dim=0)  # (f_dims,)
    # done!
    return factors_mean_next_dists, factors_ave_max_dist, factors_ave_next_dist


def aggregate_measure_distances_along_factor(
        ground_truth_dataset, representation_function,
        f_idx: int, repeats: int, batch_size: int, p='fro',
):
    # repeatedly take measurements along a factor, and return all the results
    # shapes: (repeats, f_sizes[i]) & (repeats,)
    repeated_next_dists, repeated_max_dist = repeated_measure_distances_along_factor(
        ground_truth_dataset, representation_function,
        f_idx=f_idx, repeats=repeats, batch_size=batch_size, p=p,
    )
    # aggregate results
    mean_next_dists = repeated_next_dists.mean(dim=0)
    ave_max_dist = repeated_max_dist.mean(dim=0)
    # we cant get the ave next dist directly because of cycles, so
    # we remove the max dist from those before calculating the mean
    ave_next_dist = torch.topk(repeated_next_dists, k=repeated_next_dists.shape[-1] - 1, dim=-1, largest=False, sorted=False)
    ave_next_dist = ave_next_dist.values.mean(dim=0).mean(dim=0)
    # return everything!
    # mean_next_dists: (f_sizes[i],)
    # ave_max_dist: (,)
    # ave_next_dist: (,)
    return mean_next_dists, ave_max_dist, ave_next_dist


def repeated_measure_distances_along_factor(ground_truth_dataset, representation_function, f_idx: int, repeats: int, batch_size: int, p='fro'):
    repeated_next_dists = []
    repeated_max_dist = []
    # calculate values
    # TODO: repeats dont make use of the allowed batch size effectively
    #       if a factors dimensions are too small, then the remaining space is not used.
    for _ in range(repeats):
        # generate repeated factors, varying one factor over a range
        sequential_zs = encode_all_along_factor(
            ground_truth_dataset,
            representation_function,
            f_idx=f_idx,
            batch_size=batch_size,
        )
        # calculating the distances of their representations to the next values.
        next_dists = measure_distances_along_encodings(sequential_zs, p=p)
        # distance between furthest two points
        max_dist = knn(x=sequential_zs, y=sequential_zs, k=1, largest=True, p=p).values.max()
        # append to lists
        repeated_next_dists.append(next_dists)
        repeated_max_dist.append(max_dist)
    # combine everything
    repeated_next_dists = torch.stack(repeated_next_dists, dim=0)  # shape (repeats, f_sizes[i])
    repeated_max_dist = torch.stack(repeated_max_dist, dim=0)      # shape (repeats,)
    # done!
    return repeated_next_dists, repeated_max_dist


def measure_distances_along_encodings(sequential_zs, mode='next', p='fro'):
    # find the distances to the next factors: z[i] - z[i+1]  (with wraparound)
    if mode == 'next':
        return _measure_distances_along_encodings(sequential_zs, shift=-1, p=p)
    elif mode == 'prev':
        return _measure_distances_along_encodings(sequential_zs, shift=1, p=p)
    elif mode == 'ave_prev_next':
        return torch.stack([
            _measure_distances_along_encodings(sequential_zs, shift=-1, p=p),
            _measure_distances_along_encodings(sequential_zs, shift=1, p=p),
        ], dim=0).mean(dim=0)
    else:
        raise KeyError(f'invalid mode: {mode}')


def _measure_distances_along_encodings(sequential_zs, shift: int, p='fro'):
    return torch.norm(sequential_zs - torch.roll(sequential_zs, shift, dims=0), dim=-1, p=p)


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
# KNN                                                                       #
# ========================================================================= #


def knn(x, y=None, k: int=None, largest=False, p='fro'):
    # set default values
    if y is None:
        y = x
    if k is None:
        k = y.shape[0]
    if k < 0:
        k = y.shape[0] + k
    assert 0 < k <= y.shape[0]
    # check input vectors, must be array of vectors
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == y.shape[1]
    # compute distances between each and every pair
    dist_mat = sub_matrix(x, y)
    dist_mat = torch.norm(dist_mat, dim=-1, p=p)
    # return closest distances
    return torch.topk(dist_mat, k=k, dim=-1, largest=largest, sorted=True)


def sub_matrix(a, b):
    # check input sizes
    assert a.ndim == b.ndim
    assert a.shape[1:] == b.shape[1:]
    # do pairwise subtract
    result = a[:, None, ...] - b[None, :, ...]
    # check output size
    assert result.shape == (a.shape[0], b.shape[0], *a.shape[1:])
    # done
    return result


# ========================================================================= #
# Tests                                                                     #
# ========================================================================= #


def test_sub_matrix():
    a = torch.randn(5, 2, 3)
    b = torch.randn(7, 2, 3)
    # subtract all from all
    subs = sub_matrix(a, b)
    # check that its working as intended
    for i in range(a.shape[0]):
        for j in range(b.shape[0]):
            assert torch.all(torch.eq(subs[i, j], a[i] - b[j]))
    # done
    return subs


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


if __name__ == '__main__':
    import pytorch_lightning as pl
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from disent.data.groundtruth import XYObjectData
    from disent.dataset.groundtruth import GroundTruthDataset
    from disent.frameworks.vae.unsupervised import BetaVae
    from disent.metrics import metric_dci, metric_mig
    from disent.model.ae import EncoderConv64, DecoderConv64, AutoEncoder
    from disent.transform import ToStandardisedTensor
    from disent.util import is_test_run, test_run_int

    data = XYObjectData()
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
    print(colors.lRED, metric_flatness(dataset, get_repr), colors.RST, sep='')
    print(colors.lRED, metric_flatness(dataset, get_repr), colors.RST, sep='')
    print(colors.lRED, metric_flatness(dataset, get_repr), colors.RST, sep='')

    # PHASE 2, LITTLE TRAINING
    pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, gpus=1).fit(module, dataloader)
    print(colors.lRED, metric_flatness(dataset, get_repr), colors.RST, sep='')
    print(colors.lRED, metric_flatness(dataset, get_repr), colors.RST, sep='')
    print(colors.lRED, metric_flatness(dataset, get_repr), colors.RST, sep='')

    # PHASE 3, MORE TRAINING
    pl.Trainer(logger=False, checkpoint_callback=False, max_steps=2048, gpus=1).fit(module, dataloader)
    print(colors.lRED, metric_flatness(dataset, get_repr), colors.RST, sep='')
    print(colors.lRED, metric_flatness(dataset, get_repr), colors.RST, sep='')
    print(colors.lRED, metric_flatness(dataset, get_repr), colors.RST, sep='')
