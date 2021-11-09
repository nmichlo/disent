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
Implementation of the SAP score.
Based on "Variational Inference of Disentangled Latent Concepts from Unlabeled
Observations" (https://openreview.net/forum?id=H1kG7GZAW), Section 3.
"""

import logging

import numpy as np
from sklearn import svm

from disent.dataset import DisentDataset
from disent.metrics import utils


log = logging.getLogger(__name__)


# ========================================================================= #
# SAP Scores                                                                #
# ========================================================================= #


def metric_sap(
        dataset: DisentDataset,
        representation_function,
        num_train=10000,
        num_test=5000,
        batch_size=16,
        continuous_factors=False
):
    """Computes the SAP score.
    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      num_train: Number of points used for training.
      num_test: Number of points used for testing discrete variables.
      batch_size: Batch size for sampling.
      continuous_factors: Factors are continuous variable (True) or not (False).
    Returns:
      Dictionary with SAP score.
    """
    log.debug("Generating training set.")
    mus, ys = utils.generate_batch_factor_code(dataset, representation_function, num_train, batch_size)
    mus_test, ys_test = utils.generate_batch_factor_code(dataset, representation_function, num_test, batch_size)
    log.debug("Computing score matrix.")
    return _compute_sap(mus, ys, mus_test, ys_test, continuous_factors)


def _compute_sap(mus, ys, mus_test, ys_test, continuous_factors):
    """Computes score based on both training and testing codes and factors."""
    score_matrix = _compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors)
    # Score matrix should have shape [num_latents, num_factors].
    assert score_matrix.shape[0] == mus.shape[0]
    assert score_matrix.shape[1] == ys.shape[0]

    sap_score = _compute_avg_diff_top_two(score_matrix)
    log.debug("SAP score: %.2g", sap_score)
    return {
        "sap.score": sap_score
    }


def _compute_score_matrix(mus, ys, mus_test, ys_test, continuous_factors):
    """Compute score matrix as described in Section 3."""
    num_latents = mus.shape[0]
    num_factors = ys.shape[0]
    score_matrix = np.zeros([num_latents, num_factors])
    for i in range(num_latents):
        for j in range(num_factors):
            mu_i = mus[i, :]
            y_j = ys[j, :]
            if continuous_factors:
                # Attribute is considered continuous.
                cov_mu_i_y_j = np.cov(mu_i, y_j, ddof=1)
                cov_mu_y = cov_mu_i_y_j[0, 1] ** 2
                var_mu = cov_mu_i_y_j[0, 0]
                var_y = cov_mu_i_y_j[1, 1]
                if var_mu > 1e-12:
                    score_matrix[i, j] = cov_mu_y * 1. / (var_mu * var_y)
                else:
                    score_matrix[i, j] = 0.
            else:
                # Attribute is considered discrete.
                mu_i_test = mus_test[i, :]
                y_j_test = ys_test[j, :]
                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(mu_i[:, np.newaxis], y_j)
                pred = classifier.predict(mu_i_test[:, np.newaxis])
                score_matrix[i, j] = np.mean(pred == y_j_test)
    return score_matrix


def _compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])
