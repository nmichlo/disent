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

from collections import defaultdict
from pprint import pprint
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import torch

from disent.dataset.data import Cars3d64Data
from disent.dataset.data import DSpritesData
from disent.dataset.data import Shapes3dData
from disent.dataset.data import SmallNorb64Data
from disent.frameworks.helper.reconstructions import ReconLossHandler
from tqdm import tqdm

import disent.registry as R
from disent.dataset import DisentDataset
from disent.dataset.transform import ToImgTensorF32
from disent.util.function import wrapped_partial

from research.code import register_to_disent
from research.code.dataset.data import XYSquaresData
from research.code.metrics._factored_components import _compute_dists
from research.code.metrics._factored_components import _compute_scores_from_dists
from research.code.metrics._factored_components import _numpy_concat_all_dicts


_RENAME_KEYS = {
    'mse/rsame_ground_data': 'rsame_ratio (mse)',
    'mse/rcorr_ground_data': 'rank_corr (mse)',
    'mse/lcorr_ground_data': 'linear_corr (mse)',

    'mse_gau_r31_l1.0_k3969.0_norm_sum/rsame_ground_data': 'rsame_ratio (gau)',
    'mse_gau_r31_l1.0_k3969.0_norm_sum/rcorr_ground_data': 'rank_corr (gau)',
    'mse_gau_r31_l1.0_k3969.0_norm_sum/lcorr_ground_data': 'linear_corr (gau)',

    'mse_box_r31_l1.0_k3969.0_norm_sum/rsame_ground_data': 'rsame_ratio (box)',
    'mse_box_r31_l1.0_k3969.0_norm_sum/rcorr_ground_data': 'rank_corr (box)',
    'mse_box_r31_l1.0_k3969.0_norm_sum/lcorr_ground_data': 'linear_corr (box)',

    'mse_xy8_abs63_l1.0_k1.0_norm_none/rsame_ground_data': 'rsame_ratio (xy8)',
    'mse_xy8_abs63_l1.0_k1.0_norm_none/rcorr_ground_data': 'rank_corr (xy8)',
    'mse_xy8_abs63_l1.0_k1.0_norm_none/lcorr_ground_data': 'linear_corr (xy8)',

    'mse_xy8_r47_l1.0_k3969.0_norm_sum/rsame_ground_data': 'rsame_ratio (xy8,r47,OLD)',
    'mse_xy8_r47_l1.0_k3969.0_norm_sum/rcorr_ground_data': 'rank_corr (xy8,r47,OLD)',
    'mse_xy8_r47_l1.0_k3969.0_norm_sum/lcorr_ground_data': 'linear_corr (xy8,r47,OLD)',
}

# ========================================================================= #
# plot                                                                      #
# ========================================================================= #


def _n_digits(num: int):
    if num > 0:
        return int(np.log10(num) + 1)
    if num < 0:
        return int(np.log10(-num) + 2)  # add an extra 1 for the minus sign
    else:
        return 1


def _normalise_f_name_and_idx(dataset: DisentDataset, f_idx: Optional[Union[str, int]]) -> Tuple[Optional[int], str]:
    if f_idx in ('random', None):
        f_idx = None
        f_name = 'random'
    elif isinstance(f_idx, str):
        f_idx = dataset.gt_data.normalise_factor_idx(f_idx)
        f_name = dataset.gt_data.factor_names[f_idx]
    else:
        assert isinstance(f_idx, int)
        f_name = dataset.gt_data.factor_names[f_idx]
    return f_idx, f_name


@torch.no_grad()
def _compute_mean_rcorr_ground_data(
    dataset: DisentDataset,
    f_idx: Optional[Union[str, int]],
    num_samples: int,
    repeats: int,
    progress: bool = True,
    random_batch_size: int = 16,
    losses: Sequence[str] = ('mse', 'mse_box_r31_l1.0_k3969.0')
):
    f_idx, f_name = _normalise_f_name_and_idx(dataset, f_idx)
    # recon loss handlers
    recon_handlers = {loss: R.RECON_LOSSES[loss](reduction='mean').cuda() for loss in losses}
    # storage for each loss
    distance_measures: Dict[str, List[Dict[str, np.ndarray]]] = defaultdict(list)

    # repeat!
    for i in tqdm(range(repeats), desc=f'{dataset.gt_data.name}: {f_name}', disable=not progress):
        # sample random factors
        if f_idx is None:
            factors = dataset.gt_data.sample_factors(size=random_batch_size)
        else:
            factors = dataset.gt_data.sample_random_factor_traversal(f_idx=f_idx)
        # encode factors
        xs = dataset.dataset_batch_from_factors(factors, 'input').cuda()
        factors = torch.from_numpy(factors).to(torch.float32).cpu()
        # [COMPUTE SAME RATIO & CORRELATION]
        for loss in losses:
            recon_loss = recon_handlers[loss]
            computed_dists = _compute_dists(num_samples, zs_traversal=None, xs_traversal=xs, factors=factors, recon_loss_fn=recon_loss)
            distance_measures[loss].append(computed_dists)

    # concatenate all into arrays: <shape: (repeats*num,)>
    # then aggregate over first dimension: <shape: (,)>
    distance_measures: Dict[str, float] = {
        f'{loss}/{k}': v
        for loss in losses
        for k, v in _compute_scores_from_dists(_numpy_concat_all_dicts(distance_measures[loss])).items()
    }

    # done!
    return {_RENAME_KEYS.get(k, k): v for k, v in distance_measures.items()}


# ========================================================================= #
# entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    def main(mode: str = 'compare_kernels', repeats: int = 16384):
        gt_data_classes = {
          # 'XYObject':  wrapped_partial(XYObjectData),
          # 'XYBlocks':  wrapped_partial(XYBlocksData),
            'XYSquares': wrapped_partial(XYSquaresData),
            'Cars3d':    wrapped_partial(Cars3d64Data),
            'Shapes3d':  wrapped_partial(Shapes3dData),
            'SmallNorb': wrapped_partial(SmallNorb64Data),
            'DSprites':  wrapped_partial(DSpritesData),
          # 'Mpi3d':     wrapped_partial(Mpi3dData),

            'XYSquares-1-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=1, grid_size=8, no_warnings=True),
            'XYSquares-2-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=2, grid_size=8, no_warnings=True),
            'XYSquares-3-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=3, grid_size=8, no_warnings=True),
            'XYSquares-4-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=4, grid_size=8, no_warnings=True),
            'XYSquares-5-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=5, grid_size=8, no_warnings=True),
            'XYSquares-6-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=6, grid_size=8, no_warnings=True),
            'XYSquares-7-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=7, grid_size=8, no_warnings=True),
            'XYSquares-8-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=8, grid_size=8, no_warnings=True),
        }

        include_factors = {
            'XYSquares-1-8': ('x_R',),
            'XYSquares-2-8': ('x_R',),
            'XYSquares-3-8': ('x_R',),
            'XYSquares-4-8': ('x_R',),
            'XYSquares-5-8': ('x_R',),
            'XYSquares-6-8': ('x_R',),
            'XYSquares-7-8': ('x_R',),
            'XYSquares-8-8': ('x_R',),
        }

        # 16384 * 64 = 1_048_576
        num_samples = 64
        random_batch_size = 32
        progress = True
        digits = 4

        if mode == 'all_datasets':
            ORDER = [
                'linear_corr (mse)',
                'rank_corr (mse)',
                'linear_corr (box)',
                'rank_corr (box)',
            ]
            losses = ('mse', 'mse_box_r31_l1.0_k3969.0')
        elif mode == 'compare_kernels':
            gt_data_classes = {'XYSquares-8-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=8, grid_size=8, no_warnings=True)}
            ORDER = {
                'rank_corr (mse)',
                'linear_corr (mse)',
                'rank_corr (gau)',
                'linear_corr (gau)',
                'rank_corr (box)',
                'linear_corr (box)',
                'rank_corr (xy8)',
                'linear_corr (xy8)',
                # 'rank_corr (xy8,r47,OLD)',
                # 'linear_corr (xy8,r47,OLD)',
            }
            # best hparams:
            losses = (
                'mse',                                   # orig!
                'mse_gau_r31_l1.0_k3969.0_norm_sum',     # gau: ORIG
                # 'mse_gau_r63_l1.0_k3969.0_norm_sum',   # gau: NEW  (much better than r31)
                'mse_box_r31_l1.0_k3969.0_norm_sum',     # box: ORIG
                # 'mse_xy8_r47_l1.0_k3969.0_norm_sum',   # learnt: OLD -- (DO NOT USE)
                'mse_xy8_abs63_l1.0_k1.0_norm_none',     # learnt: NEW  (SAME AS: 'mse_xy8_abs63_l1.0_k1.0_norm_none','mse_xy8_abs63_norm_none','mse_xy8_abs63')
            )
        elif mode == 'box_hparams':
            ORDER, gt_data_classes = [], {'XYSquares-8-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=8, grid_size=8, no_warnings=True)}
            losses = tuple(f'mse_box_r{r}_l1.0_k{k}_norm_sum' for k in ['100.0', '396.9', '1000.0', '3969.0', '10000.0'] for r in ['15', '31', '47', '63'])
        elif mode == 'gau_hparams':
            ORDER, gt_data_classes = [], {'XYSquares-8-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=8, grid_size=8, no_warnings=True)}
            losses = tuple(f'mse_gau_r{r}_l1.0_k{k}_norm_sum' for k in ['100.0', '396.9', '1000.0', '3969.0', '10000.0'] for r in ['15', '31', '47', '63'])
        elif mode == 'xy8_hparams':
            ORDER, gt_data_classes = [], {'XYSquares-8-8': wrapped_partial(XYSquaresData, square_size=8, grid_spacing=8, grid_size=8, no_warnings=True)}
            losses = (
                'mse_xy8_abs63',                        # this should be the same as mse_xy8_abs63_l1.0_k1.0_norm_none -- USE THIS ONE!
                'mse_xy8_abs63_norm_none',              # this should be the same as mse_xy8_abs63_l1.0_k1.0_norm_none
                'mse_xy8_abs63_l1.0_k1.0_norm_none',    # this should be the same as mse_xy8_abs63_l1.0_k1.0_norm_none
                'mse_xy8_abs63_l1.0_k10.0_norm_none',
                'mse_xy8_abs63_l1.0_k100.0_norm_none',
                'mse_xy8_abs63_l1.0_k1000.0_norm_none',
                'mse_xy8_abs63_l1.0_k10000.0_norm_none',
                # make sure scaling works
                'mse_xy8_abs63_l1.0_k1.0_norm_sum',
                'mse_xy8_abs63_l1.0_k10.0_norm_sum',
                'mse_xy8_abs63_l1.0_k100.0_norm_sum',
                'mse_xy8_abs63_l1.0_k1000.0_norm_sum',
                'mse_xy8_abs63_l1.0_k10000.0_norm_sum',
            )
        else:
            raise KeyError('invalid mode')


        for name, data_cls in  gt_data_classes.items():
            dataset = DisentDataset(data_cls(), transform=ToImgTensorF32(size=64))
            # get factor names
            factor_names = tuple(include_factors.get(name, dataset.gt_data.factor_names)) + ('random',)
            # compute over each factor name
            for i, f_name in enumerate(factor_names):
                # print variables
                f_size = dataset.gt_data.factor_sizes[dataset.gt_data.normalise_factor_idx(f_name)] if (f_name != 'random') else len(dataset)
                size_len = _n_digits(len(dataset))
                name_len = max(len(s) for s in factor_names)
                # compute scores
                try:
                    scores = _compute_mean_rcorr_ground_data(dataset, f_idx=f_name, num_samples=num_samples, repeats=repeats, random_batch_size=random_batch_size, progress=progress,  losses=losses)
                    scores = {k: v for k, v in scores.items() if ('rsame_' not in k)}
                    order = (ORDER if ORDER else scores.keys())
                    pprint(scores)
                    # NORMAL
                    # print(f'[{name}] f_idx={f_name:{name_len}s} f_size={f_size:{size_len}d} {" ".join(f"{k}={v:7.5f}" for k, v in scores.items())}')
                    # LATEX HEADINGS:
                    if i == 0:
                        print(f'[{name}] Factor Name & Factor Size & {" & ".join(f"{k:{digits}s}" for k in order if k in scores)}')
                    # LATEX
                    print(f'[{name}] {f_name:{name_len}s} & {f_size:{size_len}d} & {" & ".join(f"{scores[k]:{digits}.{digits-2}f}" for k in order if k in scores)}')
                except Exception as e:
                    # NORMAL
                    # print(f'[{name}] f_idx={f_name:{name_len}s} f_size={f_size:{size_len}d} SKIPPED!')
                    # LATEX
                    print(f'[{name}] {f_name:{name_len}s} & {f_size:{size_len}d} & {" & ".join(f"N/A" for k in ORDER)}')
                    raise e
            print()

    # RUN
    register_to_disent()

    main(mode='compare_kernels', repeats=8192)
    # main(mode='all_datasets', repeats=1024)
    # main(mode='box_hparams', repeats=128)
    # main(mode='gau_hparams', repeats=128)
    # main(mode='xy8_hparams', repeats=128)



# ========================================================================= #
# Results                                                                   #
# ========================================================================= #

# [Cars3d] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [Cars3d] elevation   &     4 & 0.94 & 0.93 & 0.90 & 0.93 & 0.88 & 0.69
# [Cars3d] azimuth     &    24 & 0.65 & 0.34 & 0.30 & 0.62 & 0.25 & 0.08
# [Cars3d] object_type &   183 & 0.52 & 0.04 & 0.04 & 0.50 & 0.01 & 0.00
# [Cars3d] random      & 17568 & 0.56 & 0.13 & 0.15 & 0.54 & 0.10 & 0.04
#
# [Shapes3d] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [Shapes3d] floor_hue   &     10 & 0.82 & 0.76 & 0.62 & 0.82 & 0.74 & 0.60
# [Shapes3d] wall_hue    &     10 & 0.82 & 0.74 & 0.60 & 0.82 & 0.72 & 0.55
# [Shapes3d] object_hue  &     10 & 0.82 & 0.71 & 0.53 & 0.82 & 0.63 & 0.41
# [Shapes3d] scale       &      8 & 0.95 & 0.88 & 0.81 & 0.95 & 0.87 & 0.71
# [Shapes3d] shape       &      4 & 0.91 & 0.79 & 0.69 & 0.90 & 0.80 & 0.58
# [Shapes3d] orientation &     15 & 0.94 & 0.92 & 0.84 & 0.89 & 0.87 & 0.74
# [Shapes3d] random      & 480000 & 0.66 & 0.45 & 0.53 & 0.60 & 0.29 & 0.29
#
# [SmallNorb] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [SmallNorb] category  &     5 & 0.75 & 0.53 & 0.44 & 0.73 & 0.47 & 0.15
# [SmallNorb] instance  &     5 & 0.73 & 0.52 & 0.37 & 0.73 & 0.51 & 0.10
# [SmallNorb] elevation &     9 & 0.94 & 0.90 & 0.81 & 0.78 & 0.64 & 0.51
# [SmallNorb] rotation  &    18 & 0.61 & 0.19 & 0.12 & 0.60 & 0.21 & 0.07
# [SmallNorb] lighting  &     6 & 0.64 & 0.29 & 0.07 & 0.64 & 0.28 & 0.07
# [SmallNorb] random    & 24300 & 0.54 & 0.14 & 0.10 & 0.54 & 0.14 & 0.07
#
# [DSprites] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [DSprites] shape       &      3 & 0.83 & 0.72 & 0.66 & 0.93 & 0.87 & 0.66
# [DSprites] scale       &      6 & 0.95 & 0.95 & 0.93 & 0.94 & 0.96 & 0.84
# [DSprites] orientation &     40 & 0.60 & 0.17 & 0.13 & 0.63 & 0.21 & 0.15
# [DSprites] position_x  &     32 & 0.90 & 0.75 & 0.66 & 0.99 & 0.83 & 0.63
# [DSprites] position_y  &     32 & 0.90 & 0.75 & 0.65 & 0.99 & 0.83 & 0.63
# [DSprites] random      & 737280 & 0.64 & 0.38 & 0.43 & 0.66 & 0.36 & 0.29
#
# [XYSquares-1-8] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [XYSquares-1-8] x_R    &      8 & 1.00 & 1.00 & 1.00 & 0.97 & 0.99 & 0.98
# [XYSquares-1-8] random & 262144 & 0.90 & 0.97 & 0.98 & 0.91 & 0.98 & 0.98
#
# [XYSquares-2-8] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [XYSquares-2-8] x_R    &      8 & 0.92 & 0.99 & 0.94 & 0.96 & 0.99 & 0.99
# [XYSquares-2-8] random & 262144 & 0.77 & 0.83 & 0.85 & 0.92 & 0.99 & 0.99
#
# [XYSquares-3-8] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [XYSquares-3-8] x_R    &      8 & 0.84 & 0.95 & 0.86 & 0.96 & 0.99 & 0.99
# [XYSquares-3-8] random & 262144 & 0.68 & 0.73 & 0.75 & 0.92 & 0.99 & 0.99
#
# [XYSquares-4-8] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [XYSquares-4-8] x_R    &      8 & 0.67 & 0.85 & 0.75 & 0.96 & 0.99 & 0.99
# [XYSquares-4-8] random & 262144 & 0.47 & 0.58 & 0.67 & 0.92 & 0.99 & 0.99
#
# [XYSquares-5-8] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [XYSquares-5-8] x_R    &      8 & 0.67 & 0.85 & 0.72 & 0.95 & 0.99 & 0.99
# [XYSquares-5-8] random & 262144 & 0.47 & 0.58 & 0.64 & 0.92 & 0.98 & 0.99
#
# [XYSquares-6-8] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [XYSquares-6-8] x_R    &      8 & 0.67 & 0.85 & 0.67 & 0.96 & 0.98 & 0.98
# [XYSquares-6-8] random & 262144 & 0.47 & 0.58 & 0.61 & 0.90 & 0.97 & 0.98
#
# [XYSquares-7-8] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [XYSquares-7-8] x_R    &      8 & 0.67 & 0.85 & 0.60 & 0.96 & 0.98 & 0.97
# [XYSquares-7-8] random & 262144 & 0.47 & 0.58 & 0.59 & 0.89 & 0.96 & 0.96
#
# [XYSquares-8-8] Factor Name & Factor Size & rsame_ratio (mse) & rank_corr (mse) & linear_corr (mse) & rsame_ratio (aug) & rank_corr (aug) & linear_corr (aug)
# [XYSquares-8-8] x_R    &      8 & 0.39 & 0.58 & 0.52 & 0.95 & 0.97 & 0.96
# [XYSquares-8-8] random & 262144 & 0.21 & 0.37 & 0.55 & 0.87 & 0.94 & 0.95

# ========================================================================= #
# Results - Reformatted                                                     #
# ========================================================================= #

# [Cars3d] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [Cars3d] elevation   & 0.90 & 0.93 & 0.69 & 0.88
# [Cars3d] azimuth     & 0.30 & 0.34 & 0.08 & 0.25
# [Cars3d] object_type & 0.04 & 0.04 & 0.00 & 0.01
# [Cars3d] random      & 0.15 & 0.13 & 0.04 & 0.10
#
# [Shapes3d] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [Shapes3d] floor_hue   & 0.62 & 0.76 & 0.60 & 0.74
# [Shapes3d] wall_hue    & 0.60 & 0.74 & 0.55 & 0.72
# [Shapes3d] object_hue  & 0.53 & 0.71 & 0.41 & 0.63
# [Shapes3d] scale       & 0.81 & 0.88 & 0.71 & 0.87
# [Shapes3d] shape       & 0.69 & 0.79 & 0.58 & 0.80
# [Shapes3d] orientation & 0.84 & 0.92 & 0.74 & 0.87
# [Shapes3d] random      & 0.53 & 0.45 & 0.29 & 0.29
#
# [SmallNorb] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [SmallNorb] category  & 0.44 & 0.53 & 0.15 & 0.47
# [SmallNorb] instance  & 0.37 & 0.52 & 0.10 & 0.51
# [SmallNorb] elevation & 0.81 & 0.90 & 0.51 & 0.64
# [SmallNorb] rotation  & 0.12 & 0.19 & 0.07 & 0.21
# [SmallNorb] lighting  & 0.07 & 0.29 & 0.07 & 0.28
# [SmallNorb] random    & 0.10 & 0.14 & 0.07 & 0.14
#
# [DSprites] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [DSprites] shape       & 0.66 & 0.72 & 0.66 & 0.87
# [DSprites] scale       & 0.93 & 0.95 & 0.84 & 0.96
# [DSprites] orientation & 0.13 & 0.17 & 0.15 & 0.21
# [DSprites] position_x  & 0.66 & 0.75 & 0.63 & 0.83
# [DSprites] position_y  & 0.65 & 0.75 & 0.63 & 0.83
# [DSprites] random      & 0.43 & 0.38 & 0.29 & 0.36
#
# [XYSquares-1-8] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [XYSquares-1-8] x_R    & 1.00 & 1.00 & 0.98 & 0.99
# [XYSquares-1-8] random & 0.98 & 0.97 & 0.98 & 0.98
#
# [XYSquares-2-8] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [XYSquares-2-8] x_R    & 0.94 & 0.99 & 0.99 & 0.99
# [XYSquares-2-8] random & 0.85 & 0.83 & 0.99 & 0.99
#
# [XYSquares-3-8] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [XYSquares-3-8] x_R    & 0.86 & 0.95 & 0.99 & 0.99
# [XYSquares-3-8] random & 0.75 & 0.73 & 0.99 & 0.99
#
# [XYSquares-4-8] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [XYSquares-4-8] x_R    & 0.75 & 0.85 & 0.99 & 0.99
# [XYSquares-4-8] random & 0.67 & 0.58 & 0.99 & 0.99
#
# [XYSquares-5-8] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [XYSquares-5-8] x_R    & 0.72 & 0.85 & 0.99 & 0.99
# [XYSquares-5-8] random & 0.64 & 0.58 & 0.99 & 0.98
#
# [XYSquares-6-8] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [XYSquares-6-8] x_R    & 0.67 & 0.85 & 0.98 & 0.98
# [XYSquares-6-8] random & 0.61 & 0.58 & 0.98 & 0.97
#
# [XYSquares-7-8] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [XYSquares-7-8] x_R    & 0.60 & 0.85 & 0.97 & 0.98
# [XYSquares-7-8] random & 0.59 & 0.58 & 0.96 & 0.96
#
# [XYSquares-8-8] Factor Name & linear_corr (mse) & rank_corr (mse) & linear_corr (aug) & rank_corr (aug)
# [XYSquares-8-8] x_R    & 0.52 & 0.58 & 0.96 & 0.97
# [XYSquares-8-8] random & 0.55 & 0.37 & 0.95 & 0.94

# ========================================================================= #
# COMPARE KERNELS -- NEW                                                    #
# ========================================================================= #

# [XYSquares-8-8] Factor Name & Factor Size & linear_corr (mse) & rank_corr (box) & rank_corr (xy8) & rank_corr (mse) & rank_corr (gau) & linear_corr (box) & linear_corr (xy8) & linear_corr (gau)
# [XYSquares-8-8] x_R    &      8 & 0.52 & 0.97 & 0.99 & 0.58 & 0.88 & 0.96 & 1.00 & 0.84
# [XYSquares-8-8] random & 262144 & 0.55 & 0.94 & 1.00 & 0.36 & 0.61 & 0.95 & 1.00 & 0.73

# MANUAL EDIT

# Loss Name        & Linear Corr. (factor) & Rank Corr. (factor) & Linear Corr. (random) & Rank Corr. (random)
# MSE              &                  0.52 &                0.58 &                  0.55 &                0.36
# MSE (Gau-Kernel) &                  0.84 &                0.88 &                  0.73 &                0.61
# MSE (Box-Kernel) &                  0.96 &                0.97 &                  0.95 &                0.94
# MSE (XY8-Kernel) &                  1.00 &                0.99 &                  1.00 &                1.00

# x_r = {
#  'linear_corr (mse)': 0.522849733858152,
#  'linear_corr (gau)': 0.8425233444009568,
#  'linear_corr (box)': 0.9568990533994665,
#  'linear_corr (xy8)': 0.9985841007312456,
#  'rank_corr (mse)': 0.5806083003878514,
#  'rank_corr (gau)': 0.8796127509043598,
#  'rank_corr (box)': 0.9725331862367547,
#  'rank_corr (xy8)': 0.9879068264304146,
#  }

# random = {
#  'linear_corr (mse)': 0.5549578765578173,
#  'linear_corr (gau)': 0.7252470489579737,
#  'linear_corr (box)': 0.945283121956908,
#  'linear_corr (xy8)': 0.9993413110499386,
#  'rank_corr (mse)': 0.36458667433987935,
#  'rank_corr (gau)': 0.6148508535668269,
#  'rank_corr (box)': 0.935241155400177,
#  'rank_corr (xy8)': 0.9981047204425746,
#  }

# ========================================================================= #
# HPARAMS                                                                   #
# ========================================================================= #

# SUMMARY:
# mse_box_r31_l1.0_k3969.0_norm_sum -- rcorr (x): 0.97, rcorr (ran): 0.94
# mse_gau_r31_l1.0_k3969.0_norm_sum -- rcorr (x): 0.88, rcorr (ran): 0.61
# mse_gau_r63_l1.0_k3969.0_norm_sum -- rcorr (x): 0.97, rcorr (ran): 0.87
# mse_xy8_abs63_l1.0_k1.0_norm_none -- rcorr (x): 0.99, rcorr (ran): 0.998   (same as: `mse_xy8_abs63`,`mse_xy8_abs63_norm_none`,`mse_xy8_abs63_l1.0_k1.0_norm_none`)

# x_R: (BOX)

# {'mse_box_r15_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.8465267326733271,
#  'mse_box_r15_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.9074030172085413,
#  'mse_box_r15_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.8421812408768276,
#  'mse_box_r15_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.899732706846721,
#  'mse_box_r15_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.8364461294011671,
#  'mse_box_r15_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.902459955644352,
#  'mse_box_r15_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.8387559590889287,
#  'mse_box_r15_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.8992828209108001,
#  'mse_box_r15_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.839586011817049,
#  'mse_box_r15_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.9017972035516646,
#  'mse_box_r31_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.7456610052478235,
#  'mse_box_r31_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.9711368561490288,
#  'mse_box_r31_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.952658055712391,
#  'mse_box_r31_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.9719123703169099,
#  'mse_box_r31_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.9509994505296753,
#  'mse_box_r31_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.9715980270707917,
#  'mse_box_r31_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.9185281602387867,
#  'mse_box_r31_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.9716384205664865,
#  'mse_box_r31_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.952864628493186,   # ORIG -- probs use this!
#  'mse_box_r31_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.9710944800602881,  # ORIG -- probs use this!
#  'mse_box_r47_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.5469040351447404,
#  'mse_box_r47_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.7586339061180715,
#  'mse_box_r47_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.683911153268716,
#  'mse_box_r47_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.7564522298329677,
#  'mse_box_r47_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.7984639858449594,
#  'mse_box_r47_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.764245721003377,
#  'mse_box_r47_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.6015761809920372,
#  'mse_box_r47_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.7495935051981993,
#  'mse_box_r47_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.7929270714898712,
#  'mse_box_r47_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.7542703815316855,
#  'mse_box_r63_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.5189231212192755,
#  'mse_box_r63_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.5792379669971695,
#  'mse_box_r63_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.5261779645463969,
#  'mse_box_r63_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.5860239628489328,
#  'mse_box_r63_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.524179961771539,
#  'mse_box_r63_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.5853527764946593,
#  'mse_box_r63_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.5222290384283436,
#  'mse_box_r63_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.5833024458678896,
#  'mse_box_r63_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.5165526108194486,
#  'mse_box_r63_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.5722994771405968}

# random: (BOX)

# {'mse_box_r15_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.7355645492374561,
#  'mse_box_r15_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.6461296788386194,
#  'mse_box_r15_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.7369420684985439,
#  'mse_box_r15_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.6414962868012578,
#  'mse_box_r15_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.7325644896797724,
#  'mse_box_r15_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.6399353290724973,
#  'mse_box_r15_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.730415588750465,
#  'mse_box_r15_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.6395086234771381,
#  'mse_box_r15_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.7334653235918743,
#  'mse_box_r15_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.6393942199843501,
#  'mse_box_r31_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.7743633840498417,
#  'mse_box_r31_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.9233773009051243,
#  'mse_box_r31_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.9382410582431485,
#  'mse_box_r31_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.9339837949342503,
#  'mse_box_r31_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.9471825129300452,  # BEST?
#  'mse_box_r31_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.9374386514298645,  # BEST?
#  'mse_box_r31_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.9109982140953756,
#  'mse_box_r31_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.9313172089479078,
#  'mse_box_r31_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.9450903194446578,  # ORIG -- probs use this
#  'mse_box_r31_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.9356454634461142,  # ORIG -- probs use this
#  'mse_box_r47_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.5869369771248776,
#  'mse_box_r47_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.7701305240518559,
#  'mse_box_r47_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.751677221885856,
#  'mse_box_r47_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.7715966871218836,
#  'mse_box_r47_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.8327945543924842,
#  'mse_box_r47_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.771589715309024,
#  'mse_box_r47_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.6611439615720487,
#  'mse_box_r47_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.7694208909826364,
#  'mse_box_r47_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.8455723029971326,
#  'mse_box_r47_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.7839774335207814,
#  'mse_box_r63_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.5542709153289006,
#  'mse_box_r63_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.3649117298131543,
#  'mse_box_r63_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.5527700068276091,
#  'mse_box_r63_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.36332730774450345,
#  'mse_box_r63_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.563230626775863,
#  'mse_box_r63_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.3711849918502138,
#  'mse_box_r63_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.5510390375629297,
#  'mse_box_r63_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.36126596021015955,
#  'mse_box_r63_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.5618312259023005,
#  'mse_box_r63_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.36803101695849627}

# x_R: (GAU)

# {'mse_gau_r15_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.7273482838641963,
#  'mse_gau_r15_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.7144203401218819,
#  'mse_gau_r15_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.7259637743598086,
#  'mse_gau_r15_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.7005330607398159,
#  'mse_gau_r15_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.7264476800975389,
#  'mse_gau_r15_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.7086928934712139,
#  'mse_gau_r15_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.7231276674231087,
#  'mse_gau_r15_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.7012189371608948,
#  'mse_gau_r15_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.7293317481072964,
#  'mse_gau_r15_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.7098309693263373,
#  'mse_gau_r31_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.8383434338101533,
#  'mse_gau_r31_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.880150834452838,
#  'mse_gau_r31_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.8401334274585273,
#  'mse_gau_r31_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.8792685133944187,
#  'mse_gau_r31_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.8462596990408467,
#  'mse_gau_r31_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.8823476556222172,
#  'mse_gau_r31_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.8383734716168756,
#  'mse_gau_r31_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.8765377171861533,
#  'mse_gau_r31_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.8403361636272052,  # ORIG -- probs use this?
#  'mse_gau_r31_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.8770459135570791,  # ORIG -- probs use this?
#  'mse_gau_r47_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.8965053635227883,
#  'mse_gau_r47_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.942814653185387,
#  'mse_gau_r47_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.9066975740516656,
#  'mse_gau_r47_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.9411909410031053,
#  'mse_gau_r47_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.9047843494918939,
#  'mse_gau_r47_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.9409217203670263,
#  'mse_gau_r47_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.9091836688143318,
#  'mse_gau_r47_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.9419431101438522,
#  'mse_gau_r47_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.9050831582460364,  # GOOD
#  'mse_gau_r47_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.9404859781515228,  # GOOD
#  'mse_gau_r63_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.8869088324812824,
#  'mse_gau_r63_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.9681351360110224,
#  'mse_gau_r63_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.9432046233907826,
#  'mse_gau_r63_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.9676927614060077,
#  'mse_gau_r63_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.9376714841351936,
#  'mse_gau_r63_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.9669249655522885,
#  'mse_gau_r63_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.9443273966580698,
#  'mse_gau_r63_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.967771782673114,
#  'mse_gau_r63_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.9389288363812284,  # GOOD
#  'mse_gau_r63_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.9677280298785211}  # GOOD

# random: (GAU)

# {'mse_gau_r15_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.6159990533616333,
#  'mse_gau_r15_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.3550159374761337,
#  'mse_gau_r15_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.6190840974821674,
#  'mse_gau_r15_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.3666723181778829,
#  'mse_gau_r15_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.622168309786848,
#  'mse_gau_r15_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.36697781941110424,
#  'mse_gau_r15_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.6283368419798032,
#  'mse_gau_r15_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.3819506598563719,
#  'mse_gau_r15_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.6137272718227919,
#  'mse_gau_r15_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.3670218847742282,
#  'mse_gau_r31_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.7257352691684524,
#  'mse_gau_r31_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.6202714962447266,
#  'mse_gau_r31_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.7252995840844141,
#  'mse_gau_r31_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.6254115079120555,
#  'mse_gau_r31_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.7274790716936281,
#  'mse_gau_r31_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.6198841473186878,
#  'mse_gau_r31_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.7340600310701334,
#  'mse_gau_r31_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.6167873984375287,
#  'mse_gau_r31_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.7274600006378323,  # ORIG
#  'mse_gau_r31_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.6118366263285681,  # ORIG
#  'mse_gau_r47_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.8158084932971688,
#  'mse_gau_r47_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.7739331988198449,
#  'mse_gau_r47_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.8271920992120669,
#  'mse_gau_r47_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.778873657013877,
#  'mse_gau_r47_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.8250240787862765,
#  'mse_gau_r47_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.7798942756913891,
#  'mse_gau_r47_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.825525769727308,
#  'mse_gau_r47_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.777835640836142,
#  'mse_gau_r47_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.8241864657986688,
#  'mse_gau_r47_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.7777636698190691,
#  'mse_gau_r63_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.8482895371617932,
#  'mse_gau_r63_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.8636270383153233,
#  'mse_gau_r63_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.8925220988850137,
#  'mse_gau_r63_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.8666087312519845,
#  'mse_gau_r63_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.8905601116453018,
#  'mse_gau_r63_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.8697913786570708,
#  'mse_gau_r63_l1.0_k396.9_norm_sum/lcorr_ground_data': 0.8982607936070446,
#  'mse_gau_r63_l1.0_k396.9_norm_sum/rcorr_ground_data': 0.8757921515353694,
#  'mse_gau_r63_l1.0_k3969.0_norm_sum/lcorr_ground_data': 0.891751398475015,   # GOOD -- probs use this?
#  'mse_gau_r63_l1.0_k3969.0_norm_sum/rcorr_ground_data': 0.8694677920447761}  # GOOD -- probs use this?

# x_R: (XY8)

# {'mse_xy8_abs63/lcorr_ground_data': 0.9985332226841734,
#  'mse_xy8_abs63/rcorr_ground_data': 0.987917449596426,
#  'mse_xy8_abs63_l1.0_k1.0_norm_none/lcorr_ground_data': 0.9985750634395287,
#  'mse_xy8_abs63_l1.0_k1.0_norm_none/rcorr_ground_data': 0.9877437598037485,
#  'mse_xy8_abs63_l1.0_k1.0_norm_sum/lcorr_ground_data': 0.5189493011290895,
#  'mse_xy8_abs63_l1.0_k1.0_norm_sum/rcorr_ground_data': 0.9910276792501457,
#  'mse_xy8_abs63_l1.0_k10.0_norm_none/lcorr_ground_data': 0.9999575954417714,
#  'mse_xy8_abs63_l1.0_k10.0_norm_none/rcorr_ground_data': 0.9879777880324067,
#  'mse_xy8_abs63_l1.0_k10.0_norm_sum/lcorr_ground_data': 0.5269050272967563,
#  'mse_xy8_abs63_l1.0_k10.0_norm_sum/rcorr_ground_data': 0.9884123616001718,
#  'mse_xy8_abs63_l1.0_k100.0_norm_none/lcorr_ground_data': 0.9999304232258539,
#  'mse_xy8_abs63_l1.0_k100.0_norm_none/rcorr_ground_data': 0.9881616858702648,
#  'mse_xy8_abs63_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.5288342188349279,
#  'mse_xy8_abs63_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.9883748032254788,
#  'mse_xy8_abs63_l1.0_k1000.0_norm_none/lcorr_ground_data': 0.999926314488479,
#  'mse_xy8_abs63_l1.0_k1000.0_norm_none/rcorr_ground_data': 0.9880581865880144,
#  'mse_xy8_abs63_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.5269727435607732,
#  'mse_xy8_abs63_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.9878274986638773,
#  'mse_xy8_abs63_l1.0_k10000.0_norm_none/lcorr_ground_data': 0.9999250116548125,
#  'mse_xy8_abs63_l1.0_k10000.0_norm_none/rcorr_ground_data': 0.9878312192520812,
#  'mse_xy8_abs63_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.6102514374734425,
#  'mse_xy8_abs63_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.9878245763300847,
#  'mse_xy8_abs63_norm_none/lcorr_ground_data': 0.9985781813175294,
#  'mse_xy8_abs63_norm_none/rcorr_ground_data': 0.9880679193468579}

# random: (XY8)

# {'mse_xy8_abs63/lcorr_ground_data': 0.9993431447352555,                          # SHOULD USE THIS!  (SAME)
#  'mse_xy8_abs63/rcorr_ground_data': 0.9981063918424973,                          # SHOULD USE THIS!  (SAME)
#  'mse_xy8_abs63_l1.0_k1.0_norm_none/lcorr_ground_data': 0.9993355789889566,      # SHOULD USE THIS!
#  'mse_xy8_abs63_l1.0_k1.0_norm_none/rcorr_ground_data': 0.9981534934650234,      # SHOULD USE THIS!
#  'mse_xy8_abs63_l1.0_k1.0_norm_sum/lcorr_ground_data': 0.5575269078895938,
#  'mse_xy8_abs63_l1.0_k1.0_norm_sum/rcorr_ground_data': 0.9802105251456182,
#  'mse_xy8_abs63_l1.0_k10.0_norm_none/lcorr_ground_data': 0.9999006262214326,
#  'mse_xy8_abs63_l1.0_k10.0_norm_none/rcorr_ground_data': 0.9981098636624177,
#  'mse_xy8_abs63_l1.0_k10.0_norm_sum/lcorr_ground_data': 0.5633754524830307,
#  'mse_xy8_abs63_l1.0_k10.0_norm_sum/rcorr_ground_data': 0.9790422482307086,
#  'mse_xy8_abs63_l1.0_k100.0_norm_none/lcorr_ground_data': 0.999896386933507,
#  'mse_xy8_abs63_l1.0_k100.0_norm_none/rcorr_ground_data': 0.9981110027949393,
#  'mse_xy8_abs63_l1.0_k100.0_norm_sum/lcorr_ground_data': 0.5638532721375747,
#  'mse_xy8_abs63_l1.0_k100.0_norm_sum/rcorr_ground_data': 0.9809908504654221,
#  'mse_xy8_abs63_l1.0_k1000.0_norm_none/lcorr_ground_data': 0.9998916501321604,
#  'mse_xy8_abs63_l1.0_k1000.0_norm_none/rcorr_ground_data': 0.9980985297573617,
#  'mse_xy8_abs63_l1.0_k1000.0_norm_sum/lcorr_ground_data': 0.5664912801305914,
#  'mse_xy8_abs63_l1.0_k1000.0_norm_sum/rcorr_ground_data': 0.9813511742194941,
#  'mse_xy8_abs63_l1.0_k10000.0_norm_none/lcorr_ground_data': 0.9998963655043154,
#  'mse_xy8_abs63_l1.0_k10000.0_norm_none/rcorr_ground_data': 0.9981026861256567,
#  'mse_xy8_abs63_l1.0_k10000.0_norm_sum/lcorr_ground_data': 0.6673302758366136,
#  'mse_xy8_abs63_l1.0_k10000.0_norm_sum/rcorr_ground_data': 0.9807396889725206,
#  'mse_xy8_abs63_norm_none/lcorr_ground_data': 0.9993412106533186,                # SHOULD USE THIS!  (SAME)
#  'mse_xy8_abs63_norm_none/rcorr_ground_data': 0.9981519882968966}                # SHOULD USE THIS!  (SAME)

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
