# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# CHANGES:
# - converted from tensorflow to pytorch
# - removed gin config
# - uses disent objects and classes
# - renamed functions

"""
Utility functions that are useful for the different metrics.
"""

from numbers import Number
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Optional
from typing import Protocol
from typing import TypeVar
from typing import Union

import numpy as np
import sklearn
from tqdm import tqdm

from disent.dataset import DisentDataset
from disent.util import to_numpy
from disent.util.function import wrapped_partial


# ========================================================================= #
# Metric Wrapper                                                            #
# ========================================================================= #


T = TypeVar('T')


class Metric(Generic[T]):

    def __init__(
        self,
        name: str,
        metric_fn: T,  # Callable[[...], Dict[str, Number]]
        default_kwargs: Optional[Dict[str, Any]] = None,
        fast_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self._name = name
        self._orig_fn           = metric_fn
        self._metric_fn_default = wrapped_partial(self._orig_fn, **(default_kwargs if default_kwargs else {}))
        self._metric_fn_fast    = wrapped_partial(self._orig_fn, **(fast_kwargs    if fast_kwargs    else {}))

    # How do we get a type hint for `__call__` so that its signature matches `T`?
    def __call__(self, *args, **kwargs) -> Dict[str, Number]:
        return self._metric_fn_default(*args, **kwargs)

    @property
    def compute(self) -> T:
        return self._metric_fn_default

    @property
    def compute_fast(self) -> T:
        return self._metric_fn_fast

    @property
    def unwrap(self) -> T:
        return self._orig_fn

    @property
    def name(self) -> str:
        return self._name

    def __str__(self):
        return f'metric-{self.name}'


def make_metric(
    name: str,
    default_kwargs: Optional[Dict[str, Any]] = None,
    fast_kwargs: Optional[Dict[str, Any]] = None,
) -> Callable[[T], Union[Metric[T], T]]:
    """
    Metrics should be decorated using this function to set defaults!
    Two versions of the metric should exist.
        1. Recommended settings
           - This should give reliable results, but may be very slow, multiple minutes to half an
             hour or more for some metrics depending on the underlying model, data and ground-truth factors.
        2. Faster settings
           - This should give a decent results, but should be decently fast, a few seconds/minutes at most.
             This is not used for testing
    """
    # `Union[Metric[T], T]` is hack to get type hint on `__call__`
    def _wrap_fn_as_metric(metric_fn: T) -> Union[Metric[T], T]:
        return Metric(name=name, metric_fn=metric_fn, default_kwargs=default_kwargs, fast_kwargs=fast_kwargs)
    return _wrap_fn_as_metric


# ========================================================================= #
# utils                                                                     #
# ========================================================================= #


def generate_batch_factor_code(
        dataset: DisentDataset,
        representation_function,
        num_points: int,
        batch_size: int,
        show_progress: bool = False,
):
    """Sample a single training sample based on a mini-batch of ground-truth data.
    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observation as input and outputs a representation.
      num_points: Number of points to sample.
      batch_size: Batchsize to sample points.
      show_progress: if a progress bar should be shown
    Returns:
      representations: Codes (num_codes, num_points)-np array.
      factors: Factors generating the codes (num_factors, num_points)-np array.
    """
    # TODO: this can be cleaned up and simplified
    #       maybe use chunked()
    representations = None
    factors = None
    i = 0
    with tqdm(total=num_points, disable=not show_progress) as bar:
        while i < num_points:
            num_points_iter = min(num_points - i, batch_size)
            current_observations, current_factors = dataset.dataset_sample_batch_with_factors(num_points_iter, mode='input')
            if i == 0:
                factors = current_factors
                representations = to_numpy(representation_function(current_observations))
            else:
                factors = np.vstack((factors, current_factors))
                representations = np.vstack((representations, to_numpy(representation_function(current_observations))))
            i += num_points_iter
            bar.update(num_points_iter)
    return np.transpose(representations), np.transpose(factors)


def split_train_test(observations, train_percentage):
    """
    Splits observations into a train and test set.
    Args:
      observations: Observations to split in train and test. They can be the
        representation or the observed factors of variation. The shape is
        (num_dimensions, num_points) and the split is over the points.
      train_percentage: Fraction of observations to be used for training.
    Returns:
      observations_train: Observations to be used for training.
      observations_test: Observations to be used for testing.
    """
    num_labelled_samples = observations.shape[1]
    num_labelled_samples_train = int(np.ceil(num_labelled_samples * train_percentage))
    num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
    observations_train = observations[:, :num_labelled_samples_train]
    observations_test = observations[:, num_labelled_samples_train:]
    assert observations_test.shape[1] == num_labelled_samples_test, "Wrong size of the test set."
    return observations_train, observations_test


def obtain_representation(observations, representation_function, batch_size):
    """"Obtain representations from observations.
    Args:
      observations: Observations for which we compute the representation.
      representation_function: Function that takes observation as input and
        outputs a representation.
      batch_size: Batch size to compute the representation.
    Returns:
      representations: Codes (num_codes, num_points)-Numpy array.
    """
    representations = None
    # TODO: use chunked
    num_points = observations.shape[0]
    i = 0
    while i < num_points:
        num_points_iter = min(num_points - i, batch_size)
        current_observations = observations[i:i + num_points_iter]
        if i == 0:
            representations = to_numpy(representation_function(current_observations))
        else:
            representations = np.vstack((representations, to_numpy(representation_function(current_observations))))
        i += num_points_iter
    return np.transpose(representations)


def histogram_discretize(target, num_bins=20):
    """
    Discretization based on histograms.
    """
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(target[i, :], num_bins)[1][:-1])
    return discretized


def discrete_mutual_info(mus, ys):
    """
    Compute discrete mutual information.
    """
    num_codes = mus.shape[0]
    num_factors = ys.shape[0]
    m = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            m[i, j] = sklearn.metrics.mutual_info_score(ys[j, :], mus[i, :])
    return m


def discrete_entropy(ys):
    """
    Compute discrete mutual information.
    """
    num_factors = ys.shape[0]
    h = np.zeros(num_factors)
    for j in range(num_factors):
        h[j] = sklearn.metrics.mutual_info_score(ys[j, :], ys[j, :])
    return h

# ========================================================================= #
# END                                                                       #
# ========================================================================= #
