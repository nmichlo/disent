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

from typing import Dict
from typing import List
from typing import Optional
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
    'mse.rsame_ground_data': 'rsame_ratio (mse)',
    'mse.rcorr_ground_data': 'rank_corr (mse)',
    'mse.lcorr_ground_data': 'linear_corr (mse)',

    'aug.rsame_ground_data': 'rsame_ratio (aug)',
    'aug.rcorr_ground_data': 'rank_corr (aug)',
    'aug.lcorr_ground_data': 'linear_corr (aug)',
}

ORDER = [
    'rsame_ratio (mse)',
    'rank_corr (mse)',
    'linear_corr (mse)',
    'rsame_ratio (aug)',
    'rank_corr (aug)',
    'linear_corr (aug)',
]


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
def _compute_mean_rcorr_ground_data(dataset: DisentDataset, f_idx: Optional[Union[str, int]], num_samples: int, repeats: int, progress: bool = True, random_batch_size: int = 16, enable_aug_loss: bool = True):
    f_idx, f_name = _normalise_f_name_and_idx(dataset, f_idx)

    # storage!
    distance_measures_mse: List[Dict[str, np.ndarray]] = []
    distance_measures_aug: List[Dict[str, np.ndarray]] = []

    # recon loss handlers
    mse: ReconLossHandler = R.RECON_LOSSES['mse'](reduction='mean').cuda()
    aug: ReconLossHandler = R.RECON_LOSSES['mse_box_r31_l1.0_k3969.0'](reduction='mean').cuda()

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
        computed_dists_mse = _compute_dists(num_samples, zs_traversal=None, xs_traversal=xs, factors=factors, recon_loss_fn=mse)
        distance_measures_mse.append(computed_dists_mse)
        # [COMPUTE SAME RATIO & CORRELATION] -- box blur
        if enable_aug_loss:
            computed_dists_aug = _compute_dists(num_samples, zs_traversal=None, xs_traversal=xs, factors=factors, recon_loss_fn=aug)
            distance_measures_aug.append(computed_dists_aug)

    # concatenate all into arrays: <shape: (repeats*num,)>
    # then aggregate over first dimension: <shape: (,)>
    distance_measures_mse: Dict[str, float] = {f'mse.{k}': v for k, v in _compute_scores_from_dists(_numpy_concat_all_dicts(distance_measures_mse)).items()}
    distance_measures_aug: Dict[str, float] = {f'aug.{k}': v for k, v in _compute_scores_from_dists(_numpy_concat_all_dicts(distance_measures_aug)).items()} if enable_aug_loss else {}

    # done!
    return {_RENAME_KEYS[k]: v for k, v in {**distance_measures_mse, **distance_measures_aug}.items()}


# ========================================================================= #
# entrypoint                                                                #
# ========================================================================= #


if __name__ == '__main__':

    def main():
        gt_data_classes = {
          # 'XYObject':  wrapped_partial(XYObjectData),
          # 'XYBlocks':  wrapped_partial(XYBlocksData),
          #   'XYSquares': wrapped_partial(XYSquaresData),
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
        repeats = 16384
        progress = True
        digits = 4
        enable_aug_loss = True

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
                    scores = _compute_mean_rcorr_ground_data(dataset, f_idx=f_name, num_samples=num_samples, repeats=repeats, random_batch_size=random_batch_size, progress=progress, enable_aug_loss=enable_aug_loss)
                    # NORMAL
                    # print(f'[{name}] f_idx={f_name:{name_len}s} f_size={f_size:{size_len}d} {" ".join(f"{k}={v:7.5f}" for k, v in scores.items())}')
                    # LATEX HEADINGS:
                    if i == 0:
                        print(f'[{name}] Factor Name & Factor Size & {" & ".join(f"{k:{digits}s}" for k in ORDER if k in scores)}')
                    # LATEX
                    print(f'[{name}] {f_name:{name_len}s} & {f_size:{size_len}d} & {" & ".join(f"{scores[k]:{digits}.{digits-2}f}" for k in ORDER if k in scores)}')
                except Exception as e:
                    # NORMAL
                    # print(f'[{name}] f_idx={f_name:{name_len}s} f_size={f_size:{size_len}d} SKIPPED!')
                    # LATEX
                    print(f'[{name}] {f_name:{name_len}s} & {f_size:{size_len}d} & {" & ".join(f"N/A" for k in ORDER)}')
                    raise e
            print()

    # RUN
    register_to_disent()
    main()


# ========================================================================= #
# Results                                                                   #
# ========================================================================= #

# [Cars3d] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [Cars3d] elevation   &     4 & 0.94 & 0.90 & 0.93
# [Cars3d] azimuth     &    24 & 0.65 & 0.31 & 0.34
# [Cars3d] object_type &   183 & 0.52 & 0.04 & 0.04
# [Cars3d] random      & 17568 & 0.56 & 0.15 & 0.13
#
# [Shapes3d] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [Shapes3d] floor_hue   &     10 & 0.82 & 0.63 & 0.76
# [Shapes3d] wall_hue    &     10 & 0.82 & 0.60 & 0.74
# [Shapes3d] object_hue  &     10 & 0.82 & 0.53 & 0.71
# [Shapes3d] scale       &      8 & 0.95 & 0.81 & 0.88
# [Shapes3d] shape       &      4 & 0.91 & 0.69 & 0.79
# [Shapes3d] orientation &     15 & 0.94 & 0.84 & 0.92
# [Shapes3d] random      & 480000 & 0.66 & 0.52 & 0.45
#
# [SmallNorb] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [SmallNorb] category  &     5 & 0.75 & 0.44 & 0.53
# [SmallNorb] instance  &     5 & 0.73 & 0.37 & 0.52
# [SmallNorb] elevation &     9 & 0.94 & 0.81 & 0.90
# [SmallNorb] rotation  &    18 & 0.61 & 0.12 & 0.18
# [SmallNorb] lighting  &     6 & 0.64 & 0.07 & 0.29
# [SmallNorb] random    & 24300 & 0.54 & 0.10 & 0.14
#
# [DSprites] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [DSprites] shape       &      3 & 0.83 & 0.66 & 0.72
# [DSprites] scale       &      6 & 0.95 & 0.94 & 0.95
# [DSprites] orientation &     40 & 0.60 & 0.13 & 0.17
# [DSprites] position_x  &     32 & 0.90 & 0.66 & 0.75
# [DSprites] position_y  &     32 & 0.90 & 0.66 & 0.75
# [DSprites] random      & 737280 & 0.64 & 0.43 & 0.37
#
# [XYSquares-1-8] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [XYSquares-1-8] x_R    &      8 & 1.00 & 1.00 & 1.00
# [XYSquares-1-8] random & 262144 & 0.90 & 0.98 & 0.97
#
# [XYSquares-2-8] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [XYSquares-2-8] x_R    &      8 & 0.92 & 0.94 & 0.99
# [XYSquares-2-8] random & 262144 & 0.78 & 0.85 & 0.83
#
# [XYSquares-3-8] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [XYSquares-3-8] x_R    &      8 & 0.84 & 0.86 & 0.95
# [XYSquares-3-8] random & 262144 & 0.68 & 0.75 & 0.73
#
# [XYSquares-4-8] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [XYSquares-4-8] x_R    &      8 & 0.67 & 0.75 & 0.85
# [XYSquares-4-8] random & 262144 & 0.47 & 0.67 & 0.58
#
# [XYSquares-5-8] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [XYSquares-5-8] x_R    &      8 & 0.67 & 0.72 & 0.85
# [XYSquares-5-8] random & 262144 & 0.47 & 0.64 & 0.58
#
# [XYSquares-6-8] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [XYSquares-6-8] x_R    &      8 & 0.67 & 0.67 & 0.85
# [XYSquares-6-8] random & 262144 & 0.47 & 0.61 & 0.58
#
# [XYSquares-7-8] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [XYSquares-7-8] x_R    &      8 & 0.67 & 0.60 & 0.85
# [XYSquares-7-8] random & 262144 & 0.47 & 0.58 & 0.58
#
# [XYSquares-8-8] Factor Name & Factor Size & rsame_ratio & linear_corr & rank_corr
# [XYSquares-8-8] x_R    &      8 & 0.39 & 0.52 & 0.58
# [XYSquares-8-8] random & 262144 & 0.21 & 0.56 & 0.37

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
