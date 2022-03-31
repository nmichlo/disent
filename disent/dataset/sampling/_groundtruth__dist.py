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

from fractions import Fraction
from typing import List
from typing import Optional
from typing import Union

import numpy as np

from disent.dataset.data import GroundTruthData
from disent.dataset.sampling._base import BaseDisentSampler
from disent.dataset.util.state_space import StateSpace


# ========================================================================= #
# Ground Truth Dist Sampler                                                 #
# ========================================================================= #


class GroundTruthDistSampler(BaseDisentSampler):

    def uninit_copy(self) -> 'GroundTruthDistSampler':
        return GroundTruthDistSampler(
            num_samples=self._num_samples,
            triplet_sample_mode=self._triplet_sample_mode,
            triplet_swap_chance=self._triplet_swap_chance,
        )

    def __init__(
            self,
            num_samples=1,
            triplet_sample_mode='manhattan_scaled',
            triplet_swap_chance=0.0,
    ):
        super().__init__(num_samples=num_samples)
        # checks
        assert num_samples in {1, 2, 3}, f'num_samples ({repr(num_samples)}) must be 1, 2 or 3'
        assert triplet_sample_mode in {'random', 'factors', 'manhattan', 'manhattan_scaled', 'combined', 'combined_scaled'}, f'sample_mode ({repr(triplet_sample_mode)}) must be one of {["random", "factors", "manhattan", "combined"]}'
        # save hparams
        self._num_samples = num_samples
        self._triplet_sample_mode = triplet_sample_mode
        self._triplet_swap_chance = triplet_swap_chance
        # scaled
        self._scaled = False
        if triplet_sample_mode.endswith('_scaled'):
            triplet_sample_mode = triplet_sample_mode[:-len('_scaled')]
            self._scaled = True
        # checks
        assert triplet_sample_mode in {'random', 'factors', 'manhattan', 'combined'}, 'It is a bug if this fails!'
        assert 0 <= triplet_swap_chance <= 1, 'triplet_swap_chance must be in range [0, 1]'
        # set vars
        self._sample_mode = triplet_sample_mode
        self._swap_chance = triplet_swap_chance
        # dataset variable
        self._state_space: Optional[StateSpace] = None

    def _init(self, dataset):
        assert isinstance(dataset, GroundTruthData), f'dataset must be an instance of {repr(GroundTruthData.__class__.__name__)}, got: {repr(dataset)}'
        self._state_space = dataset.state_space_copy()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling                                                              #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _sample_idx(self, idx):
        # sample indices
        indices = (idx, *np.random.randint(0, len(self._state_space), size=self._num_samples-1))
        # sort based on mode
        if self._num_samples == 3:
            a_i, p_i, n_i = self._swap_triple(indices)
            # randomly swap positive and negative
            if np.random.random() < self._swap_chance:
                indices = (a_i, n_i, p_i)
            else:
                indices = (a_i, p_i, n_i)
        # get data
        return indices

    def _swap_triple(self, indices):
        a_i, p_i, n_i = indices
        a_f, p_f, n_f = self._state_space.idx_to_pos(indices)
        # get the scale for everything
        # - range of positions is [0, f_size - 1], to scale between 0 and 1 we need to
        #   divide by (f_size - 1), but if the factor size is 1, we can't divide by zero
        #   so we make the minimum 1
        scale = np.maximum(1, self._state_space.factor_sizes - 1) if (self._scaled) else None
        # SWAP: manhattan
        if self._sample_mode == 'manhattan':
            if factor_dist(a_f, p_f, scale=scale) > factor_dist(a_f, n_f, scale=scale):
                return a_i, n_i, p_i
        # SWAP: factors
        elif self._sample_mode == 'factors':
            if factor_diff(a_f, p_f) > factor_diff(a_f, n_f):
                return a_i, n_i, p_i
        # SWAP: combined
        elif self._sample_mode == 'combined':
            if factor_diff(a_f, p_f) > factor_diff(a_f, n_f):
                return a_i, n_i, p_i
            elif factor_diff(a_f, p_f) == factor_diff(a_f, n_f):
                if factor_dist(a_f, p_f, scale=scale) > factor_dist(a_f, n_f, scale=scale):
                    return a_i, n_i, p_i
        # SWAP: random
        elif self._sample_mode != 'random':
            raise KeyError('invalid mode')
        # done!
        return indices


def factor_diff(f0: np.ndarray, f1: np.ndarray) -> int:
    # input types should be np.int64
    assert f0.dtype == f1.dtype == 'int64'
    # compute distances!
    return np.sum(f0 != f1)


# NOTE: scaling here should always be the same as `disentangle_loss`
def factor_dist(f0: np.ndarray, f1: np.ndarray, scale: np.ndarray = None) -> Union[Fraction, int]:
    # compute distances!
    if scale is None:
        # input types should all be np.int64
        assert f0.dtype == f1.dtype == 'int64', f'invalid dtypes, f0: {f0.dtype}, f1: {f1.dtype}'
        # we can simply sum if everything is already an integer
        return np.sum(np.abs(f0 - f1))
    else:
        # input types should all be np.int64
        assert f0.dtype == f1.dtype == scale.dtype == 'int64'
        # Division results in precision errors! We cannot simply sum divided values. We instead
        # store values as arbitrary precision rational numbers in the form of fractions This means
        # we do not lose precision while summing, and avoid comparison errors!
        #    - https://shlegeris.com/2018/10/23/sqrt.html
        #    - https://cstheory.stackexchange.com/a/4010
        # 1. first we need to convert numbers to python arbitrary precision values:
        f0: List[int]    = f0.tolist()
        f1: List[int]    = f1.tolist()
        scale: List[int] = scale.tolist()
        # 2. we need to sum values in the form of fractions
        total = Fraction(0)
        for y0, y1, s in zip(f0, f1, scale):
            total += Fraction(abs(y0 - y1), s)
        return total


# ========================================================================= #
# Investigation:                                                            #
# ========================================================================= #


if __name__ == '__main__':

    def main():
        from disent.dataset import DisentDataset
        from disent.dataset.data import XYObjectData
        from disent.dataset.data import XYObjectShadedData
        from disent.dataset.data import XYSquaresMinimalData
        from disent.dataset.data import Cars3d64Data
        from disent.dataset.data import Shapes3dData
        from disent.dataset.data import DSpritesData
        from disent.dataset.data import SmallNorb64Data
        from disent.util.seeds import TempNumpySeed
        from tqdm import tqdm

        repeats = 1000
        samples = 100

        # RESULTS - manhattan:
        #   cars3d:             orig_vs_divs=30.066%, orig_vs_frac=30.066%, divs_vs_frac=0.000%
        #   3dshapes:           orig_vs_divs=12.902%, orig_vs_frac=12.878%, divs_vs_frac=0.096%
        #   dsprites:           orig_vs_divs=24.035%, orig_vs_frac=24.032%, divs_vs_frac=0.003%
        #   smallnorb:          orig_vs_divs=18.601%, orig_vs_frac=18.598%, divs_vs_frac=0.005%
        #   xy_squares_minimal: orig_vs_divs= 1.389%, orig_vs_frac= 0.000%, divs_vs_frac=1.389%
        #   xy_object:          orig_vs_divs=15.520%, orig_vs_frac=15.511%, divs_vs_frac=0.029%
        #   xy_object:          orig_vs_divs=23.973%, orig_vs_frac=23.957%, divs_vs_frac=0.082%
        # RESULTS - combined:
        #   cars3d:             orig_vs_divs=15.428%, orig_vs_frac=15.428%, divs_vs_frac=0.000%
        #   3dshapes:           orig_vs_divs=4.982%,  orig_vs_frac= 4.968%, divs_vs_frac=0.050%
        #   dsprites:           orig_vs_divs=8.366%,  orig_vs_frac= 8.363%, divs_vs_frac=0.003%
        #   smallnorb:          orig_vs_divs=7.359%,  orig_vs_frac= 7.359%, divs_vs_frac=0.000%
        #   xy_squares_minimal: orig_vs_divs=0.610%,  orig_vs_frac= 0.000%, divs_vs_frac=0.610%
        #   xy_object:          orig_vs_divs=7.622%,  orig_vs_frac= 7.614%, divs_vs_frac=0.020%
        #   xy_object:          orig_vs_divs=8.741%,  orig_vs_frac= 8.733%, divs_vs_frac=0.046%
        for mode in ['manhattan', 'combined']:
            for data_cls in [
                Cars3d64Data,
                Shapes3dData,
                DSpritesData,
                SmallNorb64Data,
                XYSquaresMinimalData,
                XYObjectData,
                XYObjectShadedData,
            ]:
                data = data_cls()
                dataset_orig = DisentDataset(data, sampler=GroundTruthDistSampler(3, f'{mode}'))
                dataset_frac = DisentDataset(data, sampler=GroundTruthDistSampler(3, f'{mode}_scaled'))
                dataset_divs = DisentDataset(data, sampler=GroundTruthDistSampler(3, f'{mode}_scaled_INVALID'))
                # calculate the average number of mismatches between sampling methods!
                all_wrong_frac = []  # frac vs orig
                all_wrong_divs = []  # divs vs orig
                all_wrong_diff = []  # frac vs divs
                with TempNumpySeed(777):
                    progress = tqdm(range(repeats), desc=f'{mode} {data.name}')
                    for i in progress:
                        batch_seed = np.random.randint(0, 2**32)
                        with TempNumpySeed(batch_seed): idxs_orig = np.array([dataset_orig.sampler.sample(np.random.randint(0, len(dataset_orig))) for _ in range(samples)])
                        with TempNumpySeed(batch_seed): idxs_frac = np.array([dataset_frac.sampler.sample(np.random.randint(0, len(dataset_frac))) for _ in range(samples)])
                        with TempNumpySeed(batch_seed): idxs_divs = np.array([dataset_divs.sampler.sample(np.random.randint(0, len(dataset_divs))) for _ in range(samples)])
                        # check number of miss_matches
                        all_wrong_frac.append(np.sum(np.any(idxs_orig != idxs_frac, axis=-1)) / samples * 100)
                        all_wrong_divs.append(np.sum(np.any(idxs_orig != idxs_divs, axis=-1)) / samples * 100)
                        all_wrong_diff.append(np.sum(np.any(idxs_frac != idxs_divs, axis=-1)) / samples * 100)
                        # update progress bar
                        progress.set_postfix({
                            'orig_vs_divs': f'{np.mean(all_wrong_divs):5.3f}%',
                            'orig_vs_frac': f'{np.mean(all_wrong_frac):5.3f}%',
                            'divs_vs_frac': f'{np.mean(all_wrong_diff):5.3f}%',
                        })
    main()


# ========================================================================= #
# END:                                                                      #
# ========================================================================= #
