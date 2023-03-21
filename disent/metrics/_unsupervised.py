# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
# https://github.com/google-research/disentanglement_lib
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
Unsupervised scores based on code covariance and mutual information.
"""

import logging

import numpy as np
import scipy

from disent.dataset import DisentDataset
from disent.metrics import utils
from disent.metrics.utils import make_metric

log = logging.getLogger(__name__)


# ========================================================================= #
# Unsupervised Scores                                                       #
# ========================================================================= #


@make_metric("unsupervised", fast_kwargs=dict(num_train=2000))
def metric_unsupervised(dataset: DisentDataset, representation_function, num_train=10000, batch_size=16):
    """Computes unsupervised scores based on covariance and mutual information.
    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      random_state: Numpy random state used for randomness.
      artifact_dir: Optional path to directory where artifacts can be saved.
      num_train: Number of points used for training.
      batch_size: Batch size for sampling.
    Returns:
      Dictionary with scores.
    """
    log.debug("Generating training set.")
    mus_train, _ = utils.generate_batch_factor_code(dataset, representation_function, num_train, batch_size)
    num_codes = mus_train.shape[0]
    cov_mus = np.cov(mus_train)
    assert num_codes == cov_mus.shape[0]

    # Gaussian total correlation.
    gaussian_total_correlation = _gaussian_total_correlation(cov_mus)

    # Gaussian Wasserstein correlation.
    gaussian_wasserstein_correlation = _gaussian_wasserstein_correlation(cov_mus)
    gaussian_wasserstein_correlation_norm = gaussian_wasserstein_correlation / np.sum(np.diag(cov_mus))

    # Compute average mutual information between different factors.
    mus_discrete = utils.histogram_discretize(mus_train, num_bins=20)
    mutual_info_matrix = utils.discrete_mutual_info(mus_discrete, mus_discrete)
    np.fill_diagonal(mutual_info_matrix, 0)
    mutual_info_score = np.sum(mutual_info_matrix) / (num_codes**2 - num_codes)

    return {
        "unsup.mi_score": mutual_info_score,
        "unsup.gauss_total_corr": gaussian_total_correlation,
        "unsup.gauss_wasser_corr": gaussian_wasserstein_correlation,
        "unsup.gauss_wasser_corr_norm": gaussian_wasserstein_correlation_norm,
    }


def _gaussian_total_correlation(cov):
    """Computes the total correlation of a Gaussian with covariance matrix cov.
    We use that the total correlation is the KL divergence between the Gaussian
    and the product of its marginals. By design, the means of these two Gaussians
    are zero and the covariance matrix of the second Gaussian is equal to the
    covariance matrix of the first Gaussian with off-diagonal entries set to zero.
    Args:
      cov: Numpy array with covariance matrix.
    Returns:
      Scalar with total correlation.
    """
    return 0.5 * (np.sum(np.log(np.diag(cov))) - np.linalg.slogdet(cov)[1])


def _gaussian_wasserstein_correlation(cov):
    """Wasserstein L2 distance between Gaussian and the product of its marginals.
    Args:
      cov: Numpy array with covariance matrix.
    Returns:
      Scalar with score.
    """
    sqrtm = scipy.linalg.sqrtm(cov * np.expand_dims(np.diag(cov), axis=1))
    return 2 * np.trace(cov) - 2 * np.trace(sqrtm)
