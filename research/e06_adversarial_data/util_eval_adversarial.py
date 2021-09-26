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
from numba import njit
from scipy.stats import gmean


# ========================================================================= #
# Aggregate                                                                 #
# ========================================================================= #


_NP_AGGREGATE_FNS = {
    'sum': np.sum,
    'mean': np.mean,
    'gmean': gmean,  # no negatives
    'max': lambda a, axis, dtype: np.amax(a, axis=axis),  # propagate NaNs
    'min': lambda a, axis, dtype: np.amin(a, axis=axis),  # propagate NaNs
    'std': np.std,
}


def np_aggregate(array, mode: str, axis=0, dtype=None):
    try:
        fn = _NP_AGGREGATE_FNS[mode]
    except KeyError:
        raise KeyError(f'invalid aggregate mode: {repr(mode)}, must be one of: {sorted(_NP_AGGREGATE_FNS.keys())}')
    result = fn(array, axis=axis, dtype=dtype)
    if dtype is not None:
        result = result.astype(dtype)
    return result


# ========================================================================= #
# Factor Evaluation - SLOW                                                  #
# ========================================================================= #


def eval_factor_fitness_numpy(
    individual: np.ndarray,
    f_idx: int,
    f_dist_matrices: np.ndarray,
    factor_sizes: Tuple[int, ...],
    fitness_mode: str,
    exclude_diag: bool,
) -> float:
    # generate missing mask axis
    f_mask = individual.reshape(factor_sizes)
    f_mask = np.moveaxis(f_mask, f_idx, -1)
    f_mask = f_mask[..., :, None] & f_mask[..., None, :]
    # the diagonal can change statistics
    if exclude_diag:
        diag = np.arange(f_mask.shape[-1])
        f_mask[..., diag, diag] = False
    # mask the distance array | we negate the mask so that TRUE means the item is disabled
    f_dists = np.ma.masked_where(~f_mask, f_dist_matrices)

    # get distances
    if   fitness_mode == 'range':     fitness_sparse = (np.ma.max(f_dists, axis=-1) - np.ma.min(f_dists, axis=-1)).mean()
    elif fitness_mode == 'max':       fitness_sparse = (np.ma.max(f_dists, axis=-1)).mean()
    elif fitness_mode == 'std':       fitness_sparse = (np.ma.std(f_dists, axis=-1)).mean()
    else: raise KeyError(f'invalid fitness_mode: {repr(fitness_mode)}')

    # combined scores
    return fitness_sparse


# ========================================================================= #
# Factor Evaluation - FAST                                                  #
# ========================================================================= #


@njit
def eval_factor_fitness_numba__std_nodiag(
    mask: np.ndarray,
    f_dists: np.ndarray,
):
    """
    This is about 10x faster than the built in numpy version
    """
    assert f_dists.shape == (*mask.shape, mask.shape[-1])
    # totals
    total = 0.0
    count = 0
    # iterate over values -- np.ndindex is usually quite fast
    for I in np.ndindex(mask.shape[:-1]):
        # mask is broadcast to the distance matrix
        m_row = mask[I]
        d_mat = f_dists[I]
        # handle each distance matrix -- enumerate is usually faster than range
        for i, m in enumerate(m_row):
            if not m:
                continue
            # init vars
            dists = d_mat[i]
            n = 0
            s = 0.0
            s2 = 0.0
            # handle each row -- enumerate is usually faster than range
            for j, d in enumerate(dists):
                if i == j:
                    continue
                if not m_row[j]:
                    continue
                n += 1
                s += d
                s2 += d*d
            # ^^^ END j
            # update total
            if n == 1:
                count += 1
            elif n > 1:
                mean2 = (s * s) / (n * n)
                m2 = (s2 / n)
                # is this just needed because of precision errors?
                if m2 > mean2:
                    total += np.sqrt(m2 - mean2)
                count += 1
        # ^^^ END i
    if count == 0:
        return -1
    else:
        return total / count


@njit
def eval_factor_fitness_numba__range_nodiag(
    mask: np.ndarray,
    f_dists: np.ndarray,
):
    """
    This is about 10x faster than the built in numpy version
    """
    assert f_dists.shape == (*mask.shape, mask.shape[-1])
    # totals
    total = 0.0
    count = 0
    # iterate over values -- np.ndindex is usually quite fast
    for I in np.ndindex(mask.shape[:-1]):
        # mask is broadcast to the distance matrix
        m_row = mask[I]
        d_mat = f_dists[I]
        # handle each distance matrix -- enumerate is usually faster than range
        for i, m in enumerate(m_row):
            if not m:
                continue
            # init vars
            dists = d_mat[i]
            added = False
            m = 0.0
            M = 0.0
            # handle each row -- enumerate is usually faster than range
            for j, d in enumerate(dists):
                if i == j:
                    continue
                if not m_row[j]:
                    continue
                if added:
                    if d < m: m = d
                    if d > M: M = d
                else:
                    added = True
                    m = d
                    M = d
            # ^^^ END j
            # update total
            if added:
                total += (M - m)
                count += 1
        # ^^^ END i
    if count == 0:
        return -1
    else:
        return total / count


def eval_factor_fitness_numba(
    individual: np.ndarray,
    f_idx: int,
    f_dist_matrices: np.ndarray,
    factor_sizes: Tuple[int, ...],
    fitness_mode: str,
    exclude_diag: bool,
):
    """
    We only keep this function as a compatibility layer between:
        - eval_factor_fitness_numpy
        - eval_factor_fitness_numba__range_nodiag
    """
    assert exclude_diag, 'fast version of eval only supports `exclude_diag=True`'
    # usually a view
    mask = np.moveaxis(individual.reshape(factor_sizes), f_idx, -1)
    # call
    if fitness_mode == 'range':
        return eval_factor_fitness_numba__range_nodiag(mask=mask, f_dists=f_dist_matrices)
    elif fitness_mode == 'std':
        return eval_factor_fitness_numba__std_nodiag(mask=mask, f_dists=f_dist_matrices)
    else:
        raise KeyError(f'fast version of eval only supports `fitness_mode in ("range", "std")`, got: {repr(fitness_mode)}')


# ========================================================================= #
# Individual Evaluation                                                     #
# ========================================================================= #


def eval_individual(
    individual: np.ndarray,
    gt_dist_matrices: np.ndarray,
    factor_sizes: Tuple[int, ...],
    fitness_overlap_mode: str,
    fitness_overlap_aggregate: str,
    exclude_diag: bool,
    eval_factor_fitness_fn=eval_factor_fitness_numba,
) -> Tuple[float, float]:
    # evaluate all factors
    factor_scores = np.array([
        [eval_factor_fitness_fn(individual, f_idx, f_dist_matrices, factor_sizes=factor_sizes, fitness_mode=fitness_overlap_mode, exclude_diag=exclude_diag)]
        for f_idx, f_dist_matrices in enumerate(gt_dist_matrices)
    ])
    # aggregate
    factor_score = np_aggregate(factor_scores[:, 0], mode=fitness_overlap_aggregate, dtype='float64')
    kept_ratio   = individual.mean()
    # check values just in case something goes wrong!
    factor_score = np.nan_to_num(factor_score, nan=float('-inf'))
    kept_ratio   = np.nan_to_num(kept_ratio,   nan=float('-inf'))
    # return values!
    return float(factor_score), float(kept_ratio)


# ========================================================================= #
# Equality Checks                                                           #
# ========================================================================= #


def _check_equal(
    dataset_name: str = 'dsprites',
    fitness_mode: str = 'std',  # range, std
    n: int = 5,
):
    from research.e01_visual_overlap.util_compute_traversal_dists import cached_compute_all_factor_dist_matrices
    from timeit import timeit
    import research.util as H

    # load data
    gt_data = H.make_data(dataset_name)
    print(f'{dataset_name} {gt_data.factor_sizes} : {fitness_mode}')

    # get distances & individual
    all_dist_matrices = cached_compute_all_factor_dist_matrices(dataset_name)  # SHAPE FOR: s=factor_sizes, i=f_idx | (*s[:i], *s[i+1:], s[i], s[i])
    mask = np.random.random(len(gt_data)) < 0.5                                # SHAPE: (-1,)

    def eval_factor(eval_fn, f_idx: int):
        return eval_fn(
            individual=mask,
            f_idx=f_idx,
            f_dist_matrices=all_dist_matrices[f_idx],
            factor_sizes=gt_data.factor_sizes,
            fitness_mode=fitness_mode,
            exclude_diag=True,
        )

    def eval_all(eval_fn):
        return np.around([eval_factor(eval_fn, i) for i in range(gt_data.num_factors)], decimals=15)

    new_vals = eval_all(eval_factor_fitness_numba)
    new_time = timeit(lambda: eval_all(eval_factor_fitness_numba), number=n) / n
    print(f'- NEW {new_time:.5f}s {new_vals}')

    old_vals = eval_all(eval_factor_fitness_numpy)
    old_time = timeit(lambda: eval_all(eval_factor_fitness_numpy), number=n) / n
    print(f'- OLD {old_time:.5f}s {old_vals}')
    print(f'* speedup: {np.around(old_time/new_time, decimals=2)}x')


if __name__ == '__main__':

    for dataset_name in ['smallnorb', 'shapes3d', 'dsprites']:
        print('='*100)
        _check_equal(dataset_name, fitness_mode='std')
        print()
        # _check_equal(dataset_name, fitness_mode='range')
        # print('='*100)



# ========================================================================= #
# END                                                                       #
# ========================================================================= #
