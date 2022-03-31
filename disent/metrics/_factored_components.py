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
Factored Components Metric
- Michlo et al.
  https://github.com/nmichlo/msc-research
"""

import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
from disent.frameworks.helper.reconstructions import ReconLossHandler
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from disent.dataset import DisentDataset
from disent.metrics.utils import make_metric
from disent.nn.functional import torch_mean_generalized
from disent.nn.functional import torch_pca
from disent.nn.loss.reduction import batch_loss_reduction
from disent.util import to_numpy
from disent.metrics._flatness import encode_all_along_factor
from disent.metrics._flatness import encode_all_factors
from disent.metrics._flatness import filter_inactive_factors


log = logging.getLogger(__name__)


# ========================================================================= #
# flatness                                                                  #
# ========================================================================= #


_SAMPLES_MULTIPLIER_GLOBAL = 4
_SAMPLES_MULTIPLIER_FACTOR = 2


# ========================================================================= #
# flatness                                                                  #
# ========================================================================= #


def _metric_factored_components(
        dataset: DisentDataset,
        representation_function: callable,
        num_samples: int = 64,
        global_subset_size: int = 32,
        repeats: int = 1024,
        batch_size: int = 64,
        compute_distances: bool = True,
        compute_linearity: bool = True,
):
    """
    Michlo et al.
    https://github.com/nmichlo/msc-research

    Computes the factored components metric (ordering, linearity & axis alignment):

    # Distances:
        rcorr_factor_data:   rank correlation between ground-truth factor dists and MSE distances between data points
        rcorr_latent_data:   rank correlation between l2 latent dists           and MSE distances between data points
        rcorr_factor_latent: rank correlation between ground-truth factor dists and latent dists

        lcorr_factor_data:   linear correlation between ground-truth factor dists and MSE distances between data points
        lcorr_latent_data:   linear correlation between l2 latent dists           and MSE distances between data points
        lcorr_factor_latent: linear correlation between ground-truth factor dists and latent dists

        -- only active if `compute_swap_ratio=True`
        rsame_factor_data:   how similar ground-truth factor dists are compared to MSE distances between data points  MEAN: ((a<A)&(b<B)) | ((a==A)&(b==B)) | ((a>A)&(b>B))
        rsame_latent_data:   how similar l2 latent dists           are compared to MSE distances between data points  MEAN: ((a<A)&(b<B)) | ((a==A)&(b==B)) | ((a>A)&(b>B))
        rsame_factor_latent: how similar ground-truth factor dists are compared to latent dists                       MEAN: ((a<A)&(b<B)) | ((a==A)&(b==B)) | ((a>A)&(b>B))

        * modifiers:
            - .global | computed using random global samples
            - .factor | computed using random values along a ground-truth factor traversal
            - .l1     | computed using l1 distance
            - .l2     | computed using l2 distance -- (if an .l1 or .l2 tag is missing, then it is .l2 by default)

    # Linearity & Axis Alignment
        linear_ratio:           average (largest singular value over sum of singular values)
        axis_ratio:             average (largest std/variance over sum of std/variances)
        axis_alignment:         axis ratio is bounded by linear ratio - compute: axis / linear

        linear_ratio_ave:       [INVALID] average (largest singular value) over average (sum of singular values)
        axis_ratio_ave:         average (largest std/variance) over average (sum of std/variances)
        axis_alignment_ave:     [INVALID] axis ratio is bounded by linear ratio - compute: axis / linear

        * modifiers:
            - .var | computed using the variance
            - .std | computed using the standard deviation

    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observations as input and outputs a dim_representation sized representation for each observation.
      num_samples: How many random triplets are sampled from a single factor-traversal or global ranom mini-batch
      global_subset_size: Controls the size of the global random minibatch, for individual factors this is usually the size of the factor. Triplets are randomly sampled from this.
      repeats: how many times to repeat a traversal along each factors, these are then averaged together.
      batch_size: Batch size to process at any time while generating representations, should not effect metric results.
      compute_distances: If the distance components of the metric should be computed.
      compute_linearity: If the linearity components of the metric should be computed.
    Returns:
      Dictionary with metrics
    """
    # checks
    if not (compute_distances or compute_linearity):
        raise ValueError(f'Metric will not compute any values! At least one of: `compute_distances` or `compute_linearity` must be `True`')

    # compute actual metric values
    factor_scores, global_scores = _compute_factored_metric_components(
        dataset,
        representation_function,
        num_samples=num_samples,
        global_subset_size=global_subset_size,
        repeats=repeats,
        batch_size=batch_size,
        compute_distances=compute_distances,
        compute_linearity=compute_linearity,

    )

    # convert values from torch
    scores = {
        **global_scores,
        **factor_scores,
    }

    # sorted
    return {k: scores[k] for k in sorted(scores.keys())}


# EXPORT: metric_factored_components
metric_factored_components = make_metric(
    'factored_components',
    default_kwargs=dict(compute_distances=True, compute_linearity=True),
    fast_kwargs=dict(compute_distances=True, compute_linearity=True, repeats=128),
)(_metric_factored_components)

# EXPORT: metric_distances
metric_distances = make_metric(
    'distances',
    default_kwargs=dict(compute_distances=True, compute_linearity=False),
    fast_kwargs=dict(compute_distances=True, compute_linearity=False, repeats=128),
)(_metric_factored_components)

# EXPORT: metric_linearity
metric_linearity = make_metric(
    'linearity',
    default_kwargs=dict(compute_distances=False, compute_linearity=True),
    fast_kwargs=dict(compute_distances=False, compute_linearity=True, repeats=128),
)(_metric_factored_components)


# ========================================================================= #
# flatness                                                                  #
# ========================================================================= #


def _filtered_mean(values: torch.Tensor, p: Union[str, int], factor_sizes: Tuple[int, ...]):
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
def _compute_factored_metric_components(
        dataset: DisentDataset,
        representation_function,
        num_samples: int,
        global_subset_size: int,
        repeats: int,
        batch_size: int,
        compute_distances: bool,
        compute_linearity: bool,
) -> (dict, dict):

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # COMPUTE FOR EACH FACTOR
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    # shapes: (num_factors,)
    factor_values: Dict[str, np.ndarray] = _numpy_stack_all_dicts([
        _compute_factored_metric_components_along_factor(
            dataset,
            representation_function,
            f_idx=f_idx,
            num_samples=num_samples,
            repeats=repeats,
            batch_size=batch_size,
            compute_distances=compute_distances,
            compute_linearity=compute_linearity,
        )
        for f_idx in range(dataset.gt_data.num_factors)
    ])

    # aggregate for each factor
    # -- filter out factor dimensions that are incorrect. ie. size <= 1
    factor_scores = {
        k: float(_filtered_mean(torch.from_numpy(v), p='geometric', factor_sizes=dataset.gt_data.factor_sizes))
        for k, v in factor_values.items()
    }

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # RANDOM GLOBAL SAMPLES
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    if compute_distances:
        # storage
        distance_measures: List[Dict[str, np.ndarray]] = []

        # was: `iter_chunks(range(int(repeats * np.mean(dataset.gt_data.factor_sizes))), batch_size)`
        for _ in range(repeats):
            # sample random factors
            factors = dataset.gt_data.sample_factors(size=global_subset_size)
            # encode factors
            zs, xs = encode_all_factors(dataset, representation_function, factors, batch_size=batch_size, return_batch=True)
            zs, xs, factors = zs.cpu(), xs.cpu(), torch.from_numpy(factors).to(torch.float32)
            # [COMPUTE SAME RATIO & CORRELATION]: was `_SAMPLES_MULTIPLIER_GLOBAL*len(zs)`
            computed_dists = _compute_dists(num_triplets=num_samples, zs_traversal=zs, xs_traversal=xs, factors=factors)
            # [STORE DISTANCES]
            distance_measures.append(computed_dists)

        # [AGGREGATE]
        # concatenate all into arrays: <shape: (repeats*num_samples,)>
        # then aggregate over first dimension: <shape: (,)>
        distance_measures: Dict[str, np.ndarray] = _numpy_concat_all_dicts(distance_measures)
        distance_measures: Dict[str, float]      = _compute_scores_from_dists(distance_measures)
        distance_measures: Dict[str, float]      = {f'distances.{k}.global': v for k, v in distance_measures.items()}
    else:
        distance_measures: Dict[str, float] = {}

    # update global scores
    global_scores = distance_measures

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
# Collate Dict - Helper Functions - More specific than `default_collate()`  #
# ========================================================================= #


def _torch_concat_all_dicts(dists_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.cat([dists_dict[k] for dists_dict in dists_list], dim=0) for k in dists_list[0].keys()}


def _torch_stack_all_dicts(dists_list: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([dists_dict[k] for dists_dict in dists_list], dim=0) for k in dists_list[0].keys()}


def _numpy_concat_all_dicts(dists_list: List[Dict[str, Union[np.ndarray, float, int]]]) -> Dict[str, np.ndarray]:
    return {k: np.concatenate([dists_dict[k] for dists_dict in dists_list], axis=0) for k in dists_list[0].keys()}


def _numpy_stack_all_dicts(dists_list: List[Dict[str, Union[np.ndarray, float, int]]]) -> Dict[str, np.ndarray]:
    return {k: np.stack([dists_dict[k] for dists_dict in dists_list], axis=0) for k in dists_list[0].keys()}


# ========================================================================= #
# Distance Helper Functions                                                 #
# ========================================================================= #


def _same_mask(ap0, an0, ap1, an1):
    return ((ap0 < an0) & (ap1 < an1)) | ((ap0 == an0) & (ap1 == an1)) | ((ap0 > an0) & (ap1 > an1))


def _unswapped_ratio_torch(ap0: torch.Tensor, an0: torch.Tensor, ap1: torch.Tensor, an1: torch.Tensor) -> float:
    # values must correspond
    same_mask = _same_mask(ap0=ap0, an0=an0, ap1=ap1, an1=an1)
    # num values
    return float(same_mask.to(torch.float32).mean())


def _unswapped_ratio_numpy(ap0: np.ndarray, an0: np.ndarray, ap1: np.ndarray, an1: np.ndarray):
    # values must correspond
    same_mask = _same_mask(ap0=ap0, an0=an0, ap1=ap1, an1=an1)
    # num values
    return np.mean(same_mask, dtype='float32')


def _compute_dists(num_triplets: int, zs_traversal: Optional[torch.Tensor], xs_traversal: torch.Tensor, factors: Optional[torch.Tensor], recon_loss_fn=F.mse_loss) -> Dict[str, np.ndarray]:
    assert (factors      is None) or (len(factors)      == len(xs_traversal))
    assert (zs_traversal is None) or (len(zs_traversal) == len(xs_traversal))

    # get the recon loss function
    def _unreduced_loss(input, target):
        if isinstance(recon_loss_fn, ReconLossHandler):
            return recon_loss_fn.compute_unreduced_loss(input, target)
        else:
            return recon_loss_fn(input, target, reduction='none')

    # compute!
    with torch.no_grad():
        # generate random triplets
        # - {p, n} indices do not need to be sorted like triplets, these can be random.
        #   This metric is symmetric for swapped p & n values.
        idxs_a, idxs_p, idxs_n = torch.randint(0, len(xs_traversal), size=(3, num_triplets), device=xs_traversal.device)
        # compute distances -- shape: (num,)
        distances = {
            'ap_ground_dists': (torch.norm(factors[idxs_a, :] - factors[idxs_p, :], p=1, dim=-1) if (factors is not None) else torch.abs(idxs_a - idxs_p)).cpu().numpy(),
            'an_ground_dists': (torch.norm(factors[idxs_a, :] - factors[idxs_n, :], p=1, dim=-1) if (factors is not None) else torch.abs(idxs_a - idxs_n)).cpu().numpy(),
            'ap_data_dists':   batch_loss_reduction(_unreduced_loss(xs_traversal[idxs_a, ...], xs_traversal[idxs_p, ...]), reduction_dtype=torch.float32, reduction='mean').cpu().numpy(),
            'an_data_dists':   batch_loss_reduction(_unreduced_loss(xs_traversal[idxs_a, ...], xs_traversal[idxs_n, ...]), reduction_dtype=torch.float32, reduction='mean').cpu().numpy(),
        }
        # compute distances -- shape: (num,)
        if zs_traversal is not None:
            distances.update({
                'ap_latent_dists.l1': torch.norm(zs_traversal[idxs_a, :] - zs_traversal[idxs_p, :], dim=-1, p=1).cpu().numpy(),
                'an_latent_dists.l1': torch.norm(zs_traversal[idxs_a, :] - zs_traversal[idxs_n, :], dim=-1, p=1).cpu().numpy(),
                'ap_latent_dists.l2': torch.norm(zs_traversal[idxs_a, :] - zs_traversal[idxs_p, :], dim=-1, p=2).cpu().numpy(),
                'an_latent_dists.l2': torch.norm(zs_traversal[idxs_a, :] - zs_traversal[idxs_n, :], dim=-1, p=2).cpu().numpy(),
            })
        # return values -- shape: (num,)
        return distances


def _compute_scores_from_dists(dists: Dict[str, np.array]) -> Dict[str, float]:
    # [DATA & GROUND DISTS]:
    # extract the distances -- shape: (num,)
    ap_ground_dists    = dists['ap_ground_dists']
    an_ground_dists    = dists['an_ground_dists']
    ap_data_dists      = dists['ap_data_dists']
    an_data_dists      = dists['an_data_dists']
    # concatenate values -- shape: (2 * num,)
    ground_dists    = np.concatenate([ap_ground_dists,    an_ground_dists],    axis=0)
    data_dists      = np.concatenate([ap_data_dists,      an_data_dists],      axis=0)
    # compute the scores
    # - check the number of swapped elements along a factor for random triplets.
    # - compute the spearman rank correlation coefficient over the concatenated distances
    # - compute the pearman correlation coefficient over the concatenated distances
    scores = {
        'rsame_ground_data': _unswapped_ratio_numpy(ap0=ap_ground_dists, an0=an_ground_dists, ap1=ap_data_dists, an1=an_data_dists), # simplifies to: (ap_data_dists > an_data_dists).to(torch.float32).mean()
        'rcorr_ground_data': spearmanr(ground_dists, data_dists)[0],
        'lcorr_ground_data': pearsonr(ground_dists, data_dists)[0],
    }

    # [RETURN EARLY]:
    if 'ap_latent_dists.l1' not in dists:
        return scores

    # [LATENT DISTS]:
    # extract the distances -- shape: (num,)
    ap_latent_dists_l1 = dists['ap_latent_dists.l1']
    an_latent_dists_l1 = dists['an_latent_dists.l1']
    ap_latent_dists_l2 = dists['ap_latent_dists.l2']
    an_latent_dists_l2 = dists['an_latent_dists.l2']
    # concatenate values -- shape: (2 * num,)
    latent_dists_l1 = np.concatenate([ap_latent_dists_l1, an_latent_dists_l1], axis=0)
    latent_dists_l2 = np.concatenate([ap_latent_dists_l2, an_latent_dists_l2], axis=0)
    # compute the scores
    scores.update({
        # - check the number of swapped elements along a factor for random triplets.
        'rsame_ground_latent.l1': _unswapped_ratio_numpy(ap0=ap_ground_dists,    an0=an_ground_dists,    ap1=ap_latent_dists_l1, an1=an_latent_dists_l1),  # simplifies to: (ap_latent_dists > an_latent_dists).to(torch.float32).mean()
        'rsame_latent_data.l1':   _unswapped_ratio_numpy(ap0=ap_latent_dists_l1, an0=an_latent_dists_l1, ap1=ap_data_dists,      an1=an_data_dists),
        'rsame_ground_latent.l2': _unswapped_ratio_numpy(ap0=ap_ground_dists,    an0=an_ground_dists,    ap1=ap_latent_dists_l2, an1=an_latent_dists_l2),  # simplifies to: (ap_latent_dists > an_latent_dists).to(torch.float32).mean()
        'rsame_latent_data.l2':   _unswapped_ratio_numpy(ap0=ap_latent_dists_l2, an0=an_latent_dists_l2, ap1=ap_data_dists,      an1=an_data_dists),
        # - compute the spearman rank correlation coefficient over the concatenated distances
        'rcorr_ground_latent.l1': spearmanr(ground_dists,    latent_dists_l1)[0],
        'rcorr_latent_data.l1':   spearmanr(latent_dists_l1, data_dists)[0],
        'rcorr_ground_latent.l2': spearmanr(ground_dists,    latent_dists_l2)[0],
        'rcorr_latent_data.l2':   spearmanr(latent_dists_l2, data_dists)[0],
        # - compute the pearman correlation coefficient over the concatenated distances
        'lcorr_ground_latent.l1': pearsonr(ground_dists, latent_dists_l1)[0],
        'lcorr_latent_data.l1':   pearsonr(latent_dists_l1, data_dists)[0],
        'lcorr_ground_latent.l2': pearsonr(ground_dists, latent_dists_l2)[0],
        'lcorr_latent_data.l2':   pearsonr(latent_dists_l2, data_dists)[0],
    })

    # [DONE]
    return scores


# ========================================================================= #
# TRAVERSAL FLATNESS                                                        #
# ========================================================================= #


def _compute_factored_metric_components_along_factor(
        dataset: DisentDataset,
        representation_function,
        f_idx: int,
        num_samples: int,
        repeats: int,
        batch_size: int,
        compute_distances: bool,
        compute_linearity: bool,
) -> Dict[str, float]:
    # NOTE: what to do if the factor size is too small?

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # FEED FORWARD, COMPUTE ALL
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    distance_measures: List[Dict[str, np.ndarray]] = []
    linear_measures: List[Dict[str, torch.Tensor]] = []

    for i in range(repeats):
        # [ENCODE TRAVERSAL]:
        # - generate repeated factors, varying one factor over the entire range
        # - shape: (factor_size, z_size)
        zs_traversal, xs_traversal = encode_all_along_factor(dataset, representation_function, f_idx=f_idx, batch_size=batch_size, return_batch=True)
        zs_traversal = zs_traversal.cpu()

        if compute_distances:
            xs_traversal = xs_traversal.cpu()
            # [COMPUTE SAME RATIO & CORRELATION] | was: `num_triplets=_SAMPLES_MULTIPLIER_FACTOR*len(zs_traversal)`
            computed_dists = _compute_dists(num_triplets=num_samples, zs_traversal=zs_traversal, xs_traversal=xs_traversal, factors=None)
            # [STORE DISTANCES]
            distance_measures.append(computed_dists)

        if compute_linearity:
            # [VARIANCE ALONG DIFFERING AXES]:
            # 1. axis: correlation with standard basis (1, 0, 0, ...), (0, 1, 0, ...), ...
            # 2. linear: correlation along arbitrary orthogonal bases
            axis_values_var = _compute_unsorted_axis_values(zs_traversal, use_std=False)      # shape: (z_size,)
            linear_values_var = _compute_unsorted_linear_values(zs_traversal, use_std=False)  # shape: (z_size,)
            # [COMPUTE LINEARITY SCORES]:
            axis_ratio_var = _score_from_unsorted(axis_values_var, top_2=False, norm=True)      # shape: ()
            linear_ratio_var = _score_from_unsorted(linear_values_var, top_2=False, norm=True)  # shape: ()
            # [STORE SCORES]
            linear_measures.append({
                'linearity.axis_ratio.var': axis_ratio_var,
                'linearity.linear_ratio.var': linear_ratio_var,
                # aggregating linear values outside this function does not make sense, values do not correspond between repeats.
                'linearity.axis_alignment.var': axis_ratio_var / (linear_ratio_var + 1e-20),
                # temp values
                '_TEMP_.axis_values.var': axis_values_var,
            })

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # AGGREGATE DATA - For each distance measure
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    if compute_distances:
        # concatenate all into arrays: <shape: (repeats*num_samples,)>
        # then aggregate over first dimension: <shape: (,)>
        distance_measures: Dict[str, np.ndarray] = _numpy_concat_all_dicts(distance_measures)
        distance_measures: Dict[str, float]      = _compute_scores_from_dists(distance_measures)
        distance_measures: Dict[str, float]      = {f'distances.{k}.factor': v for k, v in distance_measures.items()}
    else:
        distance_measures: Dict[str, float] = {}

    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #
    # AGGREGATE DATA - For each linearity measure
    # -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~- #

    if compute_linearity:
        # stack all into arrays: <shape: (repeats, ...)>
        # then aggregate over first dimension: <shape: (...)>
        # - eg: axis_ratio  (repeats,)        -> ()
        # - eg: axis_values (repeats, z_size) -> (z_size,)
        linear_measures: Dict[str, torch.Tensor] = _torch_stack_all_dicts(linear_measures)
        linear_measures: Dict[str, torch.Tensor] = {k: v.mean(dim=0) for k, v in linear_measures.items()}
        # compute average scores & remove keys
        linear_measures['linearity.axis_ratio_ave.var'] = _score_from_unsorted(linear_measures.pop('_TEMP_.axis_values.var'), top_2=False, norm=True)  # shape: (z_size,) -> ()
        # convert values
        linear_measures: Dict[str, float] = {k: float(v) for k, v in linear_measures.items()}
    else:
        linear_measures: Dict[str, float] = {}

    # done!
    return {
        **distance_measures,
        **linear_measures,
    }


# ========================================================================= #
# END                                                                       #
# ========================================================================= #


# if __name__ == '__main__':
#
#     def main():
#         import pytorch_lightning as pl
#         from torch.utils.data import DataLoader
#         from disent.frameworks.vae import BetaVae
#         from disent.frameworks.vae import TripletVae
#         from disent.model import AutoEncoder
#         from disent.model.ae import EncoderConv64, DecoderConv64
#         from disent.util.strings import colors
#         from disent.util.profiling import Timer
#         from disent.dataset.data import XYObjectData
#         from disent.dataset.data import XYSquaresData
#         from disent.dataset.sampling import RandomSampler
#         from disent.dataset.sampling import GroundTruthDistSampler
#         from disent.dataset.transform import ToImgTensorF32
#         from disent.nn.weights import init_model_weights
#         from disent.util.seeds import seed
#         import logging
#
#         logging.basicConfig(level=logging.INFO)
#
#         def get_str(r):
#             return ', '.join(f'{k}={v:6.4f}' for k, v in r.items())
#
#         def print_r(name, steps, result, clr=colors.lYLW, t: Timer = None):
#             print(f'{clr}{name:<13} ({steps:>04}){f" {colors.GRY}[{t.pretty}]{clr}" if t else ""}: {get_str(result)}{colors.RST}')
#
#         def calculate(name, steps, dataset, get_repr):
#             with Timer() as t:
#                 r = {
#                     **metric_factored_components.compute_fast(dataset, get_repr),
#                 }
#             results.append((name, steps, r))
#             print_r(name, steps, r, colors.lRED, t=t)
#             print(colors.GRY, '='*100, colors.RST, sep='')
#             return r
#
#         class XYOverlapData(XYSquaresData):
#             def __init__(self, square_size=8, image_size=64, grid_spacing=None, num_squares=3, rgb=True):
#                 if grid_spacing is None:
#                     grid_spacing = (square_size+1) // 2
#                 super().__init__(square_size=square_size, image_size=image_size, grid_spacing=grid_spacing, num_squares=num_squares, rgb=rgb)
#
#         # datasets = [XYObjectData(rgb=False, palette='white'), XYSquaresData(), XYOverlapData(), XYObjectData()]
#         datasets = [XYObjectData()]
#
#         # TODO: fix for dead dimensions
#         # datasets = [XYObjectData(rgb=False, palette='white')]
#
#         results = []
#         for data in datasets:
#             seed(7777)
#
#             # dataset = DisentDataset(data, sampler=RandomSampler(num_samples=1), transform=ToImgTensorF32())
#             # dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, pin_memory=True)
#             # module = init_model_weights(BetaVae(
#             #     model=AutoEncoder(
#             #         encoder=EncoderConv64(x_shape=data.x_shape, z_size=9, z_multiplier=2),
#             #         decoder=DecoderConv64(x_shape=data.x_shape, z_size=9),
#             #     ),
#             #     cfg=BetaVae.cfg(beta=0.0001, loss_reduction='mean', optimizer='adam', optimizer_kwargs=dict(lr=1e-3))
#             # ), mode='xavier_normal')
#
#             dataset = DisentDataset(data, sampler=GroundTruthDistSampler(num_samples=3), transform=ToImgTensorF32())
#             dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, pin_memory=True)
#             module = init_model_weights(TripletVae(
#                 model=AutoEncoder(
#                     encoder=EncoderConv64(x_shape=data.x_shape, z_size=9, z_multiplier=2),
#                     decoder=DecoderConv64(x_shape=data.x_shape, z_size=9),
#                 ),
#                 cfg=TripletVae.cfg(beta=0.0001, loss_reduction='mean', optimizer='adam', optimizer_kwargs=dict(lr=2e-4), triplet_p=1, triplet_loss='triplet_soft', triplet_scale=1)
#             ), mode='xavier_normal')
#
#             gpus = 1 if torch.cuda.is_available() else 0
#
#             # we cannot guarantee which device the representation is on
#             get_repr = lambda x: module.encode(x.to(module.device))
#             # PHASE 1, UNTRAINED
#             pl.Trainer(logger=False, checkpoint_callback=False, fast_dev_run=True, gpus=gpus, weights_summary=None).fit(module, dataloader)
#             if torch.cuda.is_available(): module = module.to('cuda')
#             calculate(data.__class__.__name__, 0, dataset, get_repr)
#             # PHASE 2, LITTLE TRAINING
#             pl.Trainer(logger=False, checkpoint_callback=False, max_steps=256, gpus=gpus, weights_summary=None).fit(module, dataloader)
#             if torch.cuda.is_available(): module = module.to('cuda')
#             calculate(data.__class__.__name__, 256, dataset, get_repr)
#             # PHASE 3, MORE TRAINING
#             pl.Trainer(logger=False, checkpoint_callback=False, max_steps=2048, gpus=gpus, weights_summary=None).fit(module, dataloader)
#             if torch.cuda.is_available(): module = module.to('cuda')
#             calculate(data.__class__.__name__, 256+2048, dataset, get_repr)
#             results.append(None)
#
#         for result in results:
#             if result is None:
#                 print()
#                 continue
#             (name, steps, result) = result
#             print_r(name, steps, result, colors.lYLW)
#
#     main()


# if __name__ == '__main__':
#
#     def _same(ap0, an0, ap1, an1):
#         return ((ap0 < an0) & (ap1 < an1)) | ((ap0 == an0) & (ap1 == an1)) | ((ap0 > an0) & (ap1 > an1))
#
#     def main():
#         ap0 = np.array([1, 1, 1, 1,  1, 1, 1, 1,  2, 2, 2, 2,  2, 2, 2, 2])
#         an0 = np.array([1, 1, 1, 1,  2, 2, 2, 2,  1, 1, 1, 1,  2, 2, 2, 2])
#         ap1 = np.array([1, 1, 2, 2,  1, 1, 2, 2,  1, 1, 2, 2,  1, 1, 2, 2])
#         an1 = np.array([1, 2, 1, 2,  1, 2, 1, 2,  1, 2, 1, 2,  1, 2, 1, 2])
#         print(ap0)
#         print(an0)
#         print(ap1)
#         print(an1)
#
#         print(_same(ap0, an0, ap1, an1).astype('int'))
#
#     main()
