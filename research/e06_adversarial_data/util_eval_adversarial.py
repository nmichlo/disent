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
    increment_single: bool = True,
) -> float:
    assert increment_single, f'`increment_single=False` is not supported for numpy fitness evaluation'
    # generate missing mask axis
    mask = individual.reshape(factor_sizes)
    mask = np.moveaxis(mask, f_idx, -1)
    f_mask = mask[..., :, None] & mask[..., None, :]
    # the diagonal can change statistics
    if exclude_diag:
        diag = np.arange(f_mask.shape[-1])
        f_mask[..., diag, diag] = False
    # mask the distance array | we negate the mask so that TRUE means the item is disabled
    f_dists = np.ma.masked_where(~f_mask, f_dist_matrices)

    # get distances
    if   fitness_mode == 'range': agg_vals = np.ma.max(f_dists, axis=-1) - np.ma.min(f_dists, axis=-1)
    elif fitness_mode == 'max':   agg_vals = np.ma.max(f_dists, axis=-1)
    elif fitness_mode == 'std':   agg_vals = np.ma.std(f_dists, axis=-1)
    else: raise KeyError(f'invalid fitness_mode: {repr(fitness_mode)}')

    # mean -- there is still a slight difference between this version
    #         and the numba version, but this helps improve things...
    #         It might just be a precision error?
    fitness_sparse = np.ma.masked_where(~mask, agg_vals).mean()

    # combined scores
    return fitness_sparse


# ========================================================================= #
# Factor Evaluation - FAST                                                  #
# ========================================================================= #


@njit
def eval_factor_fitness_numba__std_nodiag(
    mask: np.ndarray,
    f_dists: np.ndarray,
    increment_single: bool = True
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
            # get vars
            dists = d_mat[i]
            # init vars
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
            if n > 1:
                mean2 = (s * s) / (n * n)
                m2 = (s2 / n)
                # is this just needed because of precision errors?
                if m2 > mean2:
                    total += np.sqrt(m2 - mean2)
                count += 1
            elif increment_single and (n == 1):
                total += 0.
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
    increment_single: bool = True,
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
            # get vars
            dists = d_mat[i]
            # init vars
            num_checked = False
            m = 0.0
            M = 0.0
            # handle each row -- enumerate is usually faster than range
            for j, d in enumerate(dists):
                if i == j:
                    continue
                if not m_row[j]:
                    continue
                # update range
                if num_checked > 0:
                    if d < m:
                        m = d
                    if d > M:
                        M = d
                else:
                    m = d
                    M = d
                # update num checked
                num_checked += 1
            # ^^^ END j
            # update total
            if (num_checked > 1) or (increment_single and num_checked == 1):
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
    increment_single: bool = True,
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
        return eval_factor_fitness_numba__range_nodiag(mask=mask, f_dists=f_dist_matrices, increment_single=increment_single)
    elif fitness_mode == 'std':
        return eval_factor_fitness_numba__std_nodiag(mask=mask, f_dists=f_dist_matrices, increment_single=increment_single)
    else:
        raise KeyError(f'fast version of eval only supports `fitness_mode in ("range", "std")`, got: {repr(fitness_mode)}')


# ========================================================================= #
# Individual Evaluation                                                     #
# ========================================================================= #


_EVAL_BACKENDS = {
    'numpy': eval_factor_fitness_numpy,
    'numba': eval_factor_fitness_numba,
}


def eval_individual(
    individual: np.ndarray,
    gt_dist_matrices: np.ndarray,
    factor_sizes: Tuple[int, ...],
    fitness_overlap_mode: str,
    fitness_overlap_aggregate: str,
    exclude_diag: bool,
    increment_single: bool = True,
    backend: str = 'numba',
) -> Tuple[float, float]:
    # get function
    if backend not in _EVAL_BACKENDS:
        raise KeyError(f'invalid backend: {repr(backend)}, must be one of: {sorted(_EVAL_BACKENDS.keys())}')
    eval_fn = _EVAL_BACKENDS[backend]
    # evaluate all factors
    factor_scores = np.array([
        [eval_fn(individual, f_idx, f_dist_matrices, factor_sizes=factor_sizes, fitness_mode=fitness_overlap_mode, exclude_diag=exclude_diag, increment_single=increment_single)]
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

    def eval_factor(backend: str, f_idx: int, increment_single=True):
        return _EVAL_BACKENDS[backend](
            individual=mask,
            f_idx=f_idx,
            f_dist_matrices=all_dist_matrices[f_idx],
            factor_sizes=gt_data.factor_sizes,
            fitness_mode=fitness_mode,
            exclude_diag=True,
            increment_single=increment_single,
        )

    def eval_all(backend: str, increment_single=True):
        return np.around([eval_factor(backend, i, increment_single=increment_single) for i in range(gt_data.num_factors)], decimals=15)

    new_vals = eval_all('numba', increment_single=False)
    new_time = timeit(lambda: eval_all('numba', increment_single=False), number=n) / n
    print(f'- NEW {new_time:.5f}s {new_vals} (increment_single=False)')

    new_vals = eval_all('numba')
    new_time = timeit(lambda: eval_all('numba'), number=n) / n
    print(f'- NEW {new_time:.5f}s {new_vals}')

    old_vals = eval_all('numpy')
    old_time = timeit(lambda: eval_all('numpy'), number=n) / n
    print(f'- OLD {old_time:.5f}s {old_vals}')
    print(f'* speedup: {np.around(old_time/new_time, decimals=2)}x')

    if not np.allclose(new_vals, old_vals):
        print('[WARNING]: values are not close!')


if __name__ == '__main__':

    for dataset_name in ['smallnorb', 'shapes3d', 'dsprites']:
        print('='*100)
        _check_equal(dataset_name, fitness_mode='std')
        print()
        _check_equal(dataset_name, fitness_mode='range')
        print('='*100)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
