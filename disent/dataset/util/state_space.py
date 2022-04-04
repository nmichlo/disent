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

from functools import lru_cache
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
from disent.util.iters import LengthIter
from disent.util.visualize.vis_util import get_idx_traversal


# ========================================================================= #
# Types                                                                     #
# ========================================================================= #


NonNormalisedFactorIdx  = Union[Sequence[Union[int, str]], Union[int, str]]
NonNormalisedFactorIdxs = Union[Sequence[NonNormalisedFactorIdx], NonNormalisedFactorIdx]
NonNormalisedFactors    = Union[np.ndarray, Sequence[Union[int, Sequence]]]


# ========================================================================= #
# Basic State Space                                                         #
# ========================================================================= #


class StateSpace(LengthIter):
    """
    State space where an index corresponds to coordinates (factors/positions) in the factor space.
    ie. State space with multiple factors of variation, where each factor can be a different size.
    """

    def __init__(self, factor_sizes: Sequence[int], factor_names: Optional[Sequence[str]] = None):
        super().__init__()
        # dimension: [read only]
        self.__factor_sizes = np.array(factor_sizes)
        self.__factor_sizes.flags.writeable = False
        # checks
        if self.__factor_sizes.ndim != 1:
            raise ValueError(f'`factor_sizes` must be an array with only one dimension, got shape: {self.__factor_sizes.shape}')
        if len(self.__factor_sizes) <= 0:
            raise ValueError(f'`factor_sizes` must be non-empty, got shape: {self.__factor_sizes.shape}')
        # multipliers: [read only]
        self.__factor_multipliers = _dims_multipliers(self.__factor_sizes)
        self.__factor_multipliers.flags.writeable = False
        # total permutations
        self.__size = int(np.prod(factor_sizes))
        # factor names
        self.__factor_names = tuple(f'f{i}' for i in range(self.num_factors)) if (factor_names is None) else tuple(factor_names)
        if len(self.__factor_names) != len(self.__factor_sizes):
            raise ValueError(f'Dimensionality mismatch of factor_names and factor_sizes: len({self.__factor_names}) != len({tuple(self.__factor_sizes)})')

    def __len__(self):
        """Same as self.size"""
        return self.size

    def __getitem__(self, idx):
        """Data returned based on the idx"""
        return self.idx_to_pos(idx)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Properties                                                            #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    @property
    def size(self) -> int:
        """The number of permutations of factors handled by this state space"""
        return self.__size

    @property
    def num_factors(self) -> int:
        """The number of factors handled by this state space"""
        return len(self.__factor_sizes)

    @property
    def factor_sizes(self) -> np.ndarray:
        """A list of sizes or dimensionality of factors handled by this state space"""
        return self.__factor_sizes

    @property
    def factor_names(self) -> Tuple[str, ...]:
        """A list of names of factors handled by this state space"""
        return self.__factor_names

    @property
    def factor_multipliers(self) -> np.ndarray:
        """
        The cumulative product of the factor_sizes used to convert indices to positions, and positions to indices.
        - The highest values is at the front, the lowest is at the end always being 1.
        - The size of this vector is: num_factors + 1

        Formulas:
            * Use broadcasting to get positions:
                pos = (idx[..., None] % muls[:-1]) // muls[1:]
            * Use broadcasting to get indices
                idx = np.sum(pos * muls[1:], axis=-1)
        """
        return self.__factor_multipliers

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Factor Helpers                                                        #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def normalise_factor_idx(self, factor: NonNormalisedFactorIdx) -> int:
        # convert a factor name to the factor id
        if isinstance(factor, str):
            try:
                f_idx = self.factor_names.index(factor)
            except:
                raise KeyError(f'invalid factor name: {repr(factor)} must be one of: {self.factor_names}')
        else:
            f_idx = int(factor)
        # check that the values are correct
        assert isinstance(f_idx, int)
        assert 0 <= f_idx < self.num_factors
        # return the resulting values
        return f_idx

    def normalise_factor_idxs(self, f_idxs: Optional[NonNormalisedFactorIdxs]) -> np.ndarray:
        # return the default list of factor indices
        if f_idxs is None:
            return np.arange(self.num_factors)
        # normalize a single factor into a list
        if isinstance(f_idxs, (int, str)):
            f_idxs = [f_idxs]
        # convert all the factors to their indices
        f_idxs = np.array([self.normalise_factor_idx(f_idx) for f_idx in f_idxs])
        # done! make sure there are not duplicates!
        assert len(set(f_idxs)) == len(f_idxs), 'duplicate factors were found!'
        return f_idxs

    def invert_factor_idxs(self, f_idxs: Optional[NonNormalisedFactorIdxs]) -> np.ndarray:
        f_idxs = self.normalise_factor_idxs(f_idxs)
        # create a mask of factors
        f_mask = np.ones(self.num_factors, dtype='bool')
        f_mask[f_idxs] = False
        # # # select the inverse factors
        return np.where(f_mask)[0]

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Coordinate Transform - any dim array, only last axis counts!          #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def pos_to_idx(self, positions) -> np.ndarray:
        """
        Convert a position to an index (or convert a list of positions to a list of indices)
        - positions are lists of integers, with each element < their corresponding factor size
        - indices are integers < size

        TODO: can factor_multipliers be used to speed this up?
        """
        positions = np.moveaxis(positions, source=-1, destination=0)
        return np.ravel_multi_index(positions, self.__factor_sizes)

    def idx_to_pos(self, indices) -> np.ndarray:
        """
        Convert an index to a position (or convert a list of indices to a list of positions)
        - indices are integers < size
        - positions are lists of integers, with each element < their corresponding factor size

        TODO: can factor_multipliers be used to speed this up?
        """
        positions = np.array(np.unravel_index(indices, self.__factor_sizes))
        return np.moveaxis(positions, source=0, destination=-1)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Iterators                                                             #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def iter_traversal_indices(self, f_idx: int, base_factors):
        base_factors = list(base_factors)
        base_factors[f_idx] = 0
        base_idx = self.pos_to_idx(base_factors)
        step_size = _get_step_size(tuple(self.__factor_sizes), f_idx)
        yield from range(base_idx, base_idx + step_size * self.__factor_sizes[f_idx], step_size)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # Sampling Functions - any dim array, only last axis counts!            #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def sample_indices(self, size=None):
        return np.random.randint(0, len(self), size=size)

    def sample_factors(self, size=None, f_idxs: Optional[NonNormalisedFactorIdxs] = None) -> np.ndarray:
        """
        sample randomly from all factors, otherwise the given factor_indices.
        returned values must appear in the same order as factor_indices.

        If factor factor_indices is None, all factors are sampled.
        If size=None then the array returned is the same shape as (len(factor_indices),) or factor_sizes[factor_indices]
        If size is an integer or shape, the samples returned are that shape with the last dimension
            the same size as factor_indices, ie (*size, len(factor_indices))
        """
        # get factor sizes
        if f_idxs is None:
            f_sizes = self.__factor_sizes
        else:
            f_sizes = self.__factor_sizes[self.normalise_factor_idxs(f_idxs)]  # this may be quite slow, add caching?
        # get resample size
        if size is not None:
            # empty np.array(()) gets dtype float which is incompatible with len
            size = np.append(np.array(size, dtype=int), len(f_sizes))
        # sample for factors
        return np.random.randint(0, f_sizes, size=size)

    def sample_missing_factors(self, known_factors: NonNormalisedFactors, f_idxs: NonNormalisedFactorIdxs) -> np.ndarray:
        """
        Samples the remaining factors not given in the known_factor_indices.
        ie. fills in the missing values by sampling from the unused dimensions.
        returned values are ordered by increasing factor index and not factor_indices.
        (known_factors must correspond to known_factor_indices)

        - eg. known_factors=[A], known_factor_indices=1
              BECOMES: known_factors=[A], known_factor_indices=[1]
        - eg. known_factors=[A], known_factor_indices=[1]
              = [..., A, ...]
        - eg. known_factors=[[A]], known_factor_indices=[1]
              = [[..., A, ...]]
        - eg. known_factors=[A, B], known_factor_indices=[1, 2]
              = [..., A, B, ...]
        - eg. known_factors=[[A], [B]], known_factor_indices=[1]
              = [[..., A, ...], [..., B, ...]]
        - eg. known_factors=[[A, B], [C, D]], known_factor_indices=[1, 2]
              = [[..., A, B, ...], [..., C, D, ...]]
        """
        f_idxs = self.normalise_factor_idxs(f_idxs)
        f_idxs_inv = self.invert_factor_idxs(f_idxs)
        # normalize shapes
        known_factors = np.array(known_factors)
        # checks
        assert known_factors.ndim >= 1, f'known_factors must have at least one dimension, got shape: {known_factors.shape}'
        assert known_factors.shape[-1] == len(f_idxs), f'last dimension of factors must be the same size as the number of f_idxs ({len(f_idxs)}), got shape: {known_factors.shape}'
        # replace the specified factors
        new_factors = np.empty([*known_factors.shape[:-1], self.num_factors], dtype='int')
        new_factors[..., f_idxs] = known_factors
        new_factors[..., f_idxs_inv] = self.sample_factors(size=known_factors.shape[:-1], f_idxs=f_idxs_inv)
        # done!
        return new_factors

    def resample_other_factors(self, factors: NonNormalisedFactors, f_idxs: NonNormalisedFactorIdxs) -> np.ndarray:
        """
        Resample all unspecified factors, keeping f_idxs constant.
        """
        return self.resample_given_factors(factors=factors, f_idxs=self.invert_factor_idxs(f_idxs))

    def resample_given_factors(self, factors: NonNormalisedFactors, f_idxs: NonNormalisedFactorIdxs):
        """
        Resample all specified f_idxs, keeping all remaining factors constant.
        """
        f_idxs = self.normalise_factor_idxs(f_idxs)
        new_factors = np.copy(factors)
        # checks
        assert new_factors.ndim >= 1, f'factors must have at least one dimension, got shape: {new_factors.shape}'
        assert new_factors.shape[-1] == self.num_factors, f'last dimension of factors must be the same size as the number of factors ({self.num_factors}), got shape: {new_factors.shape}'
        # replace the specified factors
        new_factors[..., f_idxs] = self.sample_factors(size=new_factors.shape[:-1], f_idxs=f_idxs)
        # done!
        return new_factors

    def _get_f_idx_and_factors_and_size(
        self,
        f_idx: Optional[int] = None,
        base_factors: Optional[NonNormalisedFactors] = None,
        num: Optional[int] = None,
    ):
        """
        :param f_idx: Sampled randomly in the range [0, num_factors) if not given.
        :param base_factors: Sampled randomly from all possible factors if not given. Coerced into the shape (1, num_factors)
        :param num: Set to the factor size `self.factor_sizes[f_idx]` if not given.
        :return: All values above in a tuple.
        """
        # choose a random factor if not given
        if f_idx is None:
            f_idx = np.random.randint(0, self.num_factors)
        # sample factors if not given
        if base_factors is None:
            base_factors = self.sample_factors(size=1)
        else:
            base_factors = np.reshape(base_factors, (1, self.num_factors))
        # get size if not given
        if num is None:
            num = self.factor_sizes[f_idx]
        else:
            assert num > 0
        # generate a traversal
        base_factors = base_factors.repeat(num, axis=0)
        # return everything
        return f_idx, base_factors, num

    def sample_random_factor_traversal(
        self,
        f_idx: Optional[int] = None,
        base_factors: Optional[NonNormalisedFactors] = None,
        num: Optional[int] = None,
        mode: str = 'interval',
        start_index: int = 0,
        return_indices: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Sample a single random factor traversal along the
        given factor index, starting from some random base sample.
        """
        f_idx, base_factors, num = self._get_f_idx_and_factors_and_size(f_idx=f_idx, base_factors=base_factors, num=num)
        # generate traversal
        base_factors[:, f_idx] = get_idx_traversal(self.factor_sizes[f_idx], num_frames=num, mode=mode, start_index=start_index)
        # return factors (num_frames, num_factors)
        if return_indices:
            return base_factors, self.pos_to_idx(base_factors)
        return base_factors

    def sample_random_factor_traversal_grid(
        self,
        num: Optional[int] = None,
        base_factors: Optional[NonNormalisedFactors] = None,
        mode: str = 'interval',
        factor_indices: Optional[NonNormalisedFactorIdxs] = None,
        return_indices: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        # default values
        if num is None:
            num = int(np.ceil(np.mean(self.factor_sizes)))
        if base_factors is None:
            base_factors = self.sample_factors()
        factor_indices = self.normalise_factor_idxs(factor_indices)
        # generate a grid of factors
        factors_grid = []
        for f_idx in factor_indices:
            factors_grid.append(self.sample_random_factor_traversal(f_idx=f_idx, base_factors=base_factors, num=num, mode=mode, start_index=0))
        factors_grid = np.stack(factors_grid, axis=0)
        # done!
        if return_indices:
            return factors_grid, self.pos_to_idx(factors_grid)
        return factors_grid


# ========================================================================= #
# Hidden State Space                                                        #
# ========================================================================= #


@lru_cache()
def _get_step_size(factor_sizes, f_idx: int):
    # check values
    assert f_idx >= 0
    assert f_idx < len(factor_sizes)
    # return values
    assert all(f > 0 for f in factor_sizes)
    # return factor size
    pos = np.zeros(len(factor_sizes), dtype='uint8')
    pos[f_idx] = 1
    return int(np.ravel_multi_index(pos, factor_sizes))


def _dims_multipliers(factor_sizes: np.ndarray) -> np.ndarray:
    factor_sizes = np.array(factor_sizes)
    assert factor_sizes.ndim == 1
    return np.append(np.cumprod(factor_sizes[::-1])[::-1], 1)


# @try_njit
# def _idx_to_pos(idxs, dims_mul):
#     factors = np.expand_dims(np.array(idxs, dtype='int'), axis=-1)
#     factors = factors % dims_mul[:-1]
#     factors //= dims_mul[1:]
#     return factors


# ========================================================================= #
# Hidden State Space                                                        #
# ========================================================================= #


# class HiddenStateSpace(_BaseStateSpace):
#
#     """
#     State space where an index corresponds to coordinates (factors/positions) in the factor space.
#     HOWEVER: some factors are treated as hidden/unknown and are thus randomly sampled.
#
#     Inputs to functions act as if known_factor_indices is the new factor space (including factor indexes).
#     Outputs from functions act in the full factor space.
#     """
#
#     def __init__(self, factor_sizes, known_factor_indices=None):
#         if known_factor_indices is None:
#             known_factor_indices = np.arange(len(factor_sizes))
#         factor_sizes, known_factor_indices = np.array(factor_sizes), np.array(known_factor_indices)
#         # known factors indices
#         self._known_factor_indices = known_factor_indices
#         self._known_factor_indices.flags.writeable = False
#         self._known_factor_sizes = factor_sizes[known_factor_indices]
#         # known state space does not include hidden variables, and assumes they are randomly sampled
#         self._known_state_space = StateSpace(self._known_factor_sizes)
#         # full state space includes hidden variables
#         self._full_state_space = StateSpace(factor_sizes)
#
#     @property
#     def size(self):
#         return self._known_state_space.size
#
#     @property
#     def num_factors(self):
#         return self._known_state_space.num_factors
#
#     @property
#     def factor_sizes(self):
#         return self._known_state_space.factor_sizes
#
#     def pos_to_idx(self, positions):
#         positions = np.array(positions)
#         assert 1 <= positions.ndim <= 2, f'positions has incorrect number of dimensions: {positions.ndim}'
#         assert positions.shape[-1] == self._full_state_space.num_factors, 'last dimension of positions must equal the full state space size'
#         # remove the unknown dimensions and return the index
#         return self._known_state_space.pos_to_idx(positions[..., self._known_factor_indices])
#
#     def idx_to_pos(self, indices):
#         indices = np.array(indices)
#         assert 0 <= indices.ndim <= 1, f'indices has incorrect number of dimensions: {indices.ndim}'
#         # get factors and return
#         known_factors = self._known_state_space.idx_to_pos(indices.reshape(-1))
#         sampled_factors = self._full_state_space.sample_missing_factors(known_factors, self._known_factor_indices)
#         return sampled_factors.reshape((*indices.shape, self._full_state_space.num_factors))
#
#     def sample_factors(self, size=None, factor_indices=None):
#         return self._full_state_space.sample_factors(
#             size,
#             self._known_factor_indices[factor_indices] if factor_indices else None
#         )
#
#     def sample_missing_factors(self, known_factors, known_factor_indices):
#         return self._full_state_space.sample_missing_factors(
#             known_factors,
#             self._known_factor_indices[known_factor_indices]
#         )
#
#     def resample_factors(self, factors, fixed_factor_indices):
#         return self._full_state_space.resample_factors(
#             factors,
#             self._known_factor_indices[fixed_factor_indices]
#         )


# ========================================================================= #
# Hidden State Space                                                        #
# ========================================================================= #

# class StateSpaceRemapIndex(object):
#     """Mapping from incorrectly ordered factors to state space indices"""
#
#     def __init__(self, factor_sizes, features):
#         self._states = StateSpace(factor_sizes)
#         # get indices of features
#         orig_indices = self._states.pos_to_idx(features)
#         if np.unique(orig_indices).size != self._states.size:
#             raise ValueError("Features do not cover the entire state space.")
#         # get indices of state space
#         state_indices = np.arange(self._states.size)
#         # mapping
#         self._state_to_orig_idx = np.zeros(self._states.size, dtype=np.int64)
#         self._state_to_orig_idx[orig_indices] = state_indices
#
#     def factors_to_orig_idx(self, factors):
#         """
#         get the original index of factors
#         """
#         return self._state_to_orig_idx[self._states.pos_to_idx(factors)]
