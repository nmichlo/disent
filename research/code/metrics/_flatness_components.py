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

"""
Flatness Metric Components
- Nathan Michlo 2021 (Unpublished)
- Cite disent
"""

import logging
from typing import Dict
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data.dataloader import default_collate

from disent.dataset import DisentDataset
from disent.nn.functional import torch_mean_generalized
from disent.nn.functional import torch_pca
from disent.nn.loss.reduction import batch_loss_reduction
from disent.util import to_numpy
from disent.util.iters import iter_chunks
from research.code.metrics._flatness import encode_all_along_factor
from research.code.metrics._flatness import encode_all_factors
from research.code.metrics._flatness import filter_inactive_factors


log = logging.getLogger(__name__)


# ========================================================================= #
# flatness                                                                  #
# ========================================================================= #


_SAMPLES_MULTIPLIER_GLOBAL = 4
_SAMPLES_MULTIPLIER_FACTOR = 2


# ========================================================================= #
# flatness                                                                  #
# ========================================================================= #


def metric_flatness_components(
        dataset: DisentDataset,
        representation_function: callable,
        factor_repeats: int = 1024,
        batch_size: int = 64,
        compute_distances: bool = True,
        compute_linearity: bool = True,
):
    """
    Computes the flatness metric components (ordering, linearity & axis alignment):

    Distances:
        rcorr_factor_data:   rank correlation between ground-truth factor dists and MSE distances between data points
        rcorr_latent_data:   rank correlation between l2 latent dists           and MSE distances between data points
        rcorr_factor_latent: rank correlation between ground-truth factor dists and l2 latent dists

        rsame_factor_data:   how similar ground-truth factor dists are compared to MSE distances between data points  MEAN: ((a<A)&(b<B)) | ((a==A)&(b==B)) | ((a>A)&(b>B))
        rsame_latent_data:   how similar l2 latent dists           are compared to MSE distances between data points  MEAN: ((a<A)&(b<B)) | ((a==A)&(b==B)) | ((a>A)&(b>B))
        rsame_factor_latent: how similar ground-truth factor dists are compared to l2 latent dists                    MEAN: ((a<A)&(b<B)) | ((a==A)&(b==B)) | ((a>A)&(b>B))

        * modifiers:
            - .global | computed using random global samples
            - .factor | computed using random values along a ground-truth factor traversal

    # Linearity & Axis Alignment
        axis_ratio:             average (largest std/variance over sum of std/variances)
        linear_ratio:           average (largest singular value over sum of singular values)
        axis_alignment:         axis ratio is bounded by linear ratio - compute: axis / linear

        ave_axis_ratio:         average (largest std/variance) over average (sum of std/variances)
        ave_linear_ratio:       [INVALID] average (largest singular value) over average (sum of singular values)
        ave_axis_alignment:     [INVALID] axis ratio is bounded by linear ratio - compute: axis / linear

        * modifiers:
            - .var | computed using the variance
            - .std | computed using the standard deviation

    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observations as input and outputs a dim_representation sized representation for each observation.
      factor_repeats: how many times to repeat a traversal along each factors, these are then averaged together.
      batch_size: Batch size to process at any time while generating representations, should not effect metric results.
      compute_distances: If the distance components of the metric should be computed.
      compute_linearity: If the linearity components of the metric should be computed.
    Returns:
      Dictionary with metrics
    """
    # checks
    if not (compute_distances or compute_linearity):
        raise ValueError(f'{metric_flatness_components.__name__} will not compute any values! At least one of: `compute_distances` or `compute_linearity` must be `True`')

    # compute actual metric values
    factor_scores, global_scores = _compute_flatness_metric_components(
        dataset,
        representation_function,
        repeats=factor_repeats,
        batch_size=batch_size,
        compute_distances=compute_distances,
        compute_linearity=compute_linearity,
    )

    # convert values from torch
    return {
        **global_scores,
        **factor_scores,
    }


def _filtered_mean(values, p, factor_sizes):
    # increase precision
    values = values.to(torch.float64)
    # check size
    assert values.shape == (len(factor_sizes),)
    # filter
    # -- filter out factor dimensions that are incorrect. ie. size <= 1
    values = filter_inactive_factors(values, factor_sizes)
    # compute mean
    mean = torch_mean_generalized(values, dim=0, p=p)
    # return decreased precision
    return to_numpy(mean.to(torch.float32))


@torch.no_grad()
def _compute_flatness_metric_components(
        dataset: DisentDataset,
        representation_function,
        repeats: int,
        batch_size: int,
        compute_distances: bool,
        compute_linearity: bool,
) -> (dict, dict):

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # COMPUTE FOR EACH FACTOR
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    factor_values = default_collate([
        _compute_flatness_metric_components_along_factor(
            dataset,
            representation_function,
            f_idx=f_idx,
            repeats=repeats,
            batch_size=batch_size,
            compute_distances=compute_distances,
            compute_linearity=compute_linearity
        )
        for f_idx in range(dataset.gt_data.num_factors)
    ])

    # aggregate for each factor
    # -- filter out factor dimensions that are incorrect. ie. size <= 1
    factor_scores = {
        k: float(_filtered_mean(v, p='geometric', factor_sizes=dataset.gt_data.factor_sizes))
        for k, v in factor_values.items()
    }

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # RANDOM GLOBAL SAMPLES
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    global_values = []
    for idxs in iter_chunks(range(int(repeats * np.mean(dataset.gt_data.factor_sizes))), batch_size):
        # sample random factors
        factors = dataset.gt_data.sample_factors(size=len(idxs))
        # encode factors
        zs, xs = encode_all_factors(dataset, representation_function, factors, batch_size=batch_size, return_batch=True)
        # [COMPUTE SAME RATIO & CORRELATION]
        computed_dists = _dists_compute_scores(_SAMPLES_MULTIPLIER_GLOBAL*len(zs), zs_traversal=zs, xs_traversal=xs, factors=torch.from_numpy(factors).to(torch.float32))
        # [UPDATE SCORES]
        global_values.append({f'distances.{k}.global': v for k, v in computed_dists.items()})

    # collect then aggregate values
    global_values = default_collate(global_values)
    global_scores = {k: float(v.mean(dim=0)) for k, v in global_values.items()}

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # RETURN
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    return factor_scores, global_scores


# ========================================================================= #
# CORE                                                                      #
# -- using variance instead of standard deviation makes it easier to        #
#    obtain high scores.                                                    #
# ========================================================================= #


def _compute_unsorted_axis_values(zs_traversal, use_std: bool = True):
    # CORRELATIONS -- SORTED IN DESCENDING ORDER:
    # correlation with standard basis (1, 0, 0, ...), (0, 1, 0, ...), ...
    axis_values = torch.var(zs_traversal, dim=0)  # (z_size,)
    if use_std:
        axis_values = torch.sqrt(axis_values)
    return axis_values


def _compute_unsorted_linear_values(zs_traversal, use_std: bool = True):
    # CORRELATIONS -- SORTED IN DESCENDING ORDER:
    # correlation along arbitrary orthogonal basis
    # -- note pca_mode='svd' returns the number of values equal to: min(factor_size, z_size)  !!! this may lower scores on average
    # -- note pca_mode='eig' returns the number of values equal to: z_size
    _, linear_values = torch_pca(zs_traversal, center=True, mode='eig')
    if use_std:
        linear_values = torch.sqrt(linear_values)
    return linear_values


def _score_from_sorted(sorted_vars: torch.Tensor, top_2: bool = False, norm: bool = True) -> torch.Tensor:
    if top_2:
        # use two max values
        # this is more like mig
        sorted_vars = sorted_vars[:2]
    # sum all values
    n = len(sorted_vars)
    r = sorted_vars[0] / torch.sum(sorted_vars)
    # get norm if needed
    if norm:
        # for: x/(x+a)
        # normalised = (x/(x+a) - (1/n)) / (1 - (1/n))
        # normalised = (x - 1/(n-1) * a) / (x + a)
        r = (r - (1/n)) / (1 - (1/n))
    # done!
    return r


def _score_from_unsorted(unsorted_values: torch.Tensor, top_2: bool = False, norm: bool = True):
    assert unsorted_values.ndim == 1
    # sort in descending order
    sorted_values = torch.sort(unsorted_values, dim=-1, descending=True).values
    # compute score
    return _score_from_sorted(sorted_values, top_2=top_2, norm=norm)


def compute_axis_score(zs_traversal: torch.Tensor, use_std: bool = True, top_2: bool = False, norm: bool = True):
    unsorted_values = _compute_unsorted_axis_values(zs_traversal, use_std=use_std)
    score = _score_from_unsorted(unsorted_values, top_2=top_2, norm=norm)
    return score


def compute_linear_score(zs_traversal: torch.Tensor, use_std: bool = True, top_2: bool = False, norm: bool = True):
    unsorted_values = _compute_unsorted_linear_values(zs_traversal, use_std=use_std)
    score = _score_from_unsorted(unsorted_values, top_2=top_2, norm=norm)
    return score


# ========================================================================= #
# Distance Helper Functions                                                 #
# ========================================================================= #


def _unswapped_ratio(ap0: torch.Tensor, an0: torch.Tensor, ap1: torch.Tensor, an1: torch.Tensor):
    # values must correspond
    same_mask = ((ap0 < an0) & (ap1 < an1)) | ((ap0 == an0) & (ap1 == an1)) | ((ap0 > an0) & (ap1 > an1))
    # num values
    return same_mask.to(torch.float32).mean()


def _dists_compute_scores(num_triplets: int, zs_traversal: torch.Tensor, xs_traversal: torch.Tensor, factors: Optional[torch.Tensor] = None) -> Dict[str, float]:
    # checks
    assert (len(zs_traversal) == len(xs_traversal)) and ((factors is None) or (len(factors) == len(zs_traversal)))
    # generate random triplets
    # - {p, n} indices do not need to be sorted like triplets, these can be random.
    #   This metric is symmetric for swapped p & n values.
    idxs_a, idxs_p, idxs_n = torch.randint(0, len(zs_traversal), size=(3, num_triplets))
    # compute distances -- shape: (num,)
    ap_ground_dists = torch.abs(idxs_a - idxs_p) if (factors is None) else torch.norm(factors[idxs_a, :] - factors[idxs_p, :], p=1, dim=-1)
    an_ground_dists = torch.abs(idxs_a - idxs_n) if (factors is None) else torch.norm(factors[idxs_a, :] - factors[idxs_n, :], p=1, dim=-1)
    ap_latent_dists = torch.norm(zs_traversal[idxs_a, :] - zs_traversal[idxs_p, :], dim=-1, p=2)
    an_latent_dists = torch.norm(zs_traversal[idxs_a, :] - zs_traversal[idxs_n, :], dim=-1, p=2)
    ap_data_dists   = batch_loss_reduction(F.mse_loss(xs_traversal[idxs_a, ...], xs_traversal[idxs_p, ...], reduction='none'), reduction_dtype=torch.float32, reduction='mean')
    an_data_dists   = batch_loss_reduction(F.mse_loss(xs_traversal[idxs_a, ...], xs_traversal[idxs_n, ...], reduction='none'), reduction_dtype=torch.float32, reduction='mean')
    # compute rsame scores -- shape: ()
    # - check the number of swapped elements along a factor for random triplets.
    rsame_ground_data   = _unswapped_ratio(ap0=ap_ground_dists, an0=an_ground_dists, ap1=ap_data_dists,   an1=an_data_dists)    # simplifies to: (ap_data_dists > an_data_dists).to(torch.float32).mean()
    rsame_ground_latent = _unswapped_ratio(ap0=ap_ground_dists, an0=an_ground_dists, ap1=ap_latent_dists, an1=an_latent_dists)  # simplifies to: (ap_latent_dists > an_latent_dists).to(torch.float32).mean()
    rsame_latent_data   = _unswapped_ratio(ap0=ap_latent_dists, an0=an_latent_dists, ap1=ap_data_dists,   an1=an_data_dists)
    # concatenate values -- shape: (2 * num,)
    ground_dists = torch.cat([ap_ground_dists, an_ground_dists], dim=0).numpy()
    latent_dists = torch.cat([ap_latent_dists, an_latent_dists], dim=0).numpy()
    data_dists   = torch.cat([ap_data_dists,   an_data_dists],   dim=0).numpy()
    # compute rcorr scores -- shape: ()
    # - compute the pearson rank correlation coefficient over the concatenated distances
    rcorr_ground_data, _   = spearmanr(ground_dists, data_dists)
    rcorr_ground_latent, _ = spearmanr(ground_dists, latent_dists)
    rcorr_latent_data, _   = spearmanr(latent_dists, data_dists)
    # return values -- shape: ()
    return {
        # same ratio
        'rsame_ground_data':   rsame_ground_data,
        'rsame_ground_latent': rsame_ground_latent,
        'rsame_latent_data':   rsame_latent_data,
        # correlation
        'rcorr_ground_data':   rcorr_ground_data,
        'rcorr_ground_latent': rcorr_ground_latent,
        'rcorr_latent_data':   rcorr_latent_data,
    }


# ========================================================================= #
# TRAVERSAL FLATNESS                                                        #
# ========================================================================= #


def _compute_flatness_metric_components_along_factor(
        dataset: DisentDataset,
        representation_function,
        f_idx: int,
        repeats: int,
        batch_size: int,
        compute_distances: bool,
        compute_linearity: bool,
) -> dict:
    # NOTE: what to do if the factor size is too small?

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # FEED FORWARD, COMPUTE ALL
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    measures = []
    for i in range(repeats):
        scores = {}

        # [ENCODE TRAVERSAL]:
        # - generate repeated factors, varying one factor over the entire range
        # - shape: (factor_size, z_size)
        zs_traversal, xs_traversal = encode_all_along_factor(dataset, representation_function, f_idx=f_idx, batch_size=batch_size, return_batch=True)

        if compute_distances:
            # [COMPUTE SAME RATIO & CORRELATION]
            computed_dists = _dists_compute_scores(_SAMPLES_MULTIPLIER_FACTOR*len(zs_traversal), zs_traversal=zs_traversal, xs_traversal=xs_traversal)
            # [UPDATE SCORES]
            scores.update({f'distances.{k}.factor': v for k, v in computed_dists.items()})

        if compute_linearity:
            # [VARIANCE ALONG DIFFERING AXES]:
            # 1. axis: correlation with standard basis (1, 0, 0, ...), (0, 1, 0, ...), ...
            # 2. linear: correlation along arbitrary orthogonal bases
            axis_values_var = _compute_unsorted_axis_values(zs_traversal, use_std=False)      # shape: (z_size,)
            linear_values_var = _compute_unsorted_linear_values(zs_traversal, use_std=False)  # shape: (z_size,)
            # [COMPUTE LINEARITY SCORES]:
            axis_ratio_var = _score_from_unsorted(axis_values_var, top_2=False, norm=True)      # shape: ()
            linear_ratio_var = _score_from_unsorted(linear_values_var, top_2=False, norm=True)  # shape: ()
            # [UPDATE SCORES]
            scores.update({
                'linearity.axis_ratio.var': axis_ratio_var,
                'linearity.linear_ratio.var': linear_ratio_var,
                # aggregating linear values outside this function does not make sense, values do not correspond between repeats.
                'linearity.axis_alignment.var': axis_ratio_var / (linear_ratio_var + 1e-20),
                # temp values
                '_TEMP_.axis_values.var': axis_values_var,
            })

        # [MERGE SCORES]
        measures.append(scores)

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # AGGREGATE DATA - For each distance measure
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    # stack all into tensors: <shape: (repeats, ...)>
    # then aggregate over first dimension: <shape: (...)>
    # - eg: axis_ratio  (repeats,)        -> ()
    # - eg: axis_values (repeats, z_size) -> (z_size,)
    measures = default_collate(measures)
    measures = {k: v.mean(dim=0) for k, v in measures.items()}

    # compute average scores & remove keys
    if compute_linearity:
        measures['linearity.ave_axis_ratio.var'] = _score_from_unsorted(measures.pop('_TEMP_.axis_values.var'), top_2=False, norm=True)  # shape: (z_size,) -> ()

    # done!
    return measures


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


# if __name__ == '__main__':
#     from disent.metrics._flatness import get_device
#     import pytorch_lightning as pl
#     from torch.optim import Adam
#     from torch.utils.data import DataLoader
#     from disent.frameworks.vae import BetaVae
#     from disent.model import AutoEncoder
#     from disent.model.ae import EncoderConv64, DecoderConv64
#     from disent.util.strings import colors
#     from disent.util.profiling import Timer
#     from disent.dataset.data import XYObjectData
#     from disent.dataset.data import XYSquaresData
#     from disent.dataset.sampling import RandomSampler
#     from disent.dataset.transform import ToImgTensorF32
#     from disent.nn.weights import init_model_weights
#     from disent.util.seeds import seed
#     import logging
#
#     logging.basicConfig(level=logging.INFO)
#
#     def get_str(r):
#         return ', '.join(f'{k}={v:6.4f}' for k, v in r.items())
#
#     def print_r(name, steps, result, clr=colors.lYLW, t: Timer = None):
#         print(f'{clr}{name:<13} ({steps:>04}){f" {colors.GRY}[{t.pretty}]{clr}" if t else ""}: {get_str(result)}{colors.RST}')
#
#     def calculate(name, steps, dataset, get_repr):
#         with Timer() as t:
#             r = {
#                 **metric_flatness_components(dataset, get_repr, factor_repeats=64, batch_size=64),
#                 # **metric_flatness(dataset, get_repr, factor_repeats=64, batch_size=64),
#             }
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
#     # TODO: fix for dead dimensions
#     # datasets = [XYObjectData(rgb=False, palette='white')]
#
#     results = []
#     for data in datasets:
#         seed(7777)
#
#         dataset = DisentDataset(data, sampler=RandomSampler(num_samples=1), transform=ToImgTensorF32())
#         dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, pin_memory=True)
#         module = init_model_weights(BetaVae(
#             model=AutoEncoder(
#                 encoder=EncoderConv64(x_shape=data.x_shape, z_size=9, z_multiplier=2),
#                 decoder=DecoderConv64(x_shape=data.x_shape, z_size=9),
#             ),
#             cfg=BetaVae.cfg(beta=0.0001, loss_reduction='mean', optimizer=torch.optim.Adam, optimizer_kwargs=dict(lr=1e-3))
#         ), mode='xavier_normal')
#
#         gpus = 1 if torch.cuda.is_available() else 0
#
#         # we cannot guarantee which device the representation is on
#         get_repr = lambda x: module.encode(x.to(module.device))
#         # PHASE 1, UNTRAINED
#         pl.Trainer(logger=False, checkpoint_callback=False, fast_dev_run=True, gpus=gpus, weights_summary=None).fit(module, dataloader)
#         if torch.cuda.is_available(): module = module.to('cuda')
#         calculate(data.__class__.__name__, 0, dataset, get_repr)
#         # PHASE 2, LITTLE TRAINING
#         pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, gpus=gpus, weights_summary=None).fit(module, dataloader)
#         if torch.cuda.is_available(): module = module.to('cuda')
#         calculate(data.__class__.__name__, 256, dataset, get_repr)
#         # PHASE 3, MORE TRAINING
#         pl.Trainer(logger=False, checkpoint_callback=False, max_steps=2048, gpus=gpus, weights_summary=None).fit(module, dataloader)
#         if torch.cuda.is_available(): module = module.to('cuda')
#         calculate(data.__class__.__name__, 256+2048, dataset, get_repr)
#         results.append(None)
#
#     for result in results:
#         if result is None:
#             print()
#             continue
#         (name, steps, result) = result
#         print_r(name, steps, result, colors.lYLW)
