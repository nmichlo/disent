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
Implementation of Disentanglement, Completeness and Informativeness.
Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""

import logging
from tqdm import tqdm

from disent.dataset import DisentDataset
from disent.metrics import utils
import numpy as np
import scipy
import scipy.stats


log = logging.getLogger(__name__)


# ========================================================================= #
# dci                                                                       #
# ========================================================================= #


def metric_dci(
        dataset: DisentDataset,
        representation_function: callable,
        num_train: int = 10000,
        num_test: int = 5000,
        batch_size: int = 16,
        boost_mode='sklearn',
        show_progress=False,
):
    """Computes the DCI scores according to Sec 2.
    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      num_train: Number of points used for training.
      num_test: Number of points used for testing.
      batch_size: Batch size for sampling.
      boost_mode: which boosting algorithm should be used [sklearn, xgboost, lightgbm] (this can have a significant effect on score)
      show_progress: If a tqdm progress bar should be shown
    Returns:
      Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    log.debug("Generating training set.")
    # mus_train are of shape [num_codes, num_train], while ys_train are of shape
    # [num_factors, num_train].
    mus_train, ys_train = utils.generate_batch_factor_code(dataset, representation_function, num_train, batch_size, show_progress=False)
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    mus_test, ys_test = utils.generate_batch_factor_code(dataset, representation_function, num_test, batch_size, show_progress=False)

    log.debug("Computing DCI metric.")
    scores = _compute_dci(mus_train, ys_train, mus_test, ys_test, boost_mode=boost_mode, show_progress=show_progress)

    return scores


def _compute_dci(mus_train, ys_train, mus_test, ys_test, boost_mode='sklearn', show_progress=False):
    """Computes score based on both training and testing codes and factors."""
    importance_matrix, train_err, test_err = _compute_importance_gbt(mus_train, ys_train, mus_test, ys_test, boost_mode=boost_mode, show_progress=show_progress)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    return {
        "dci.informativeness_train": train_err,
        "dci.informativeness_test": test_err,
        "dci.disentanglement": _disentanglement(importance_matrix),
        "dci.completeness": _completeness(importance_matrix),
    }


def _compute_importance_gbt(x_train, y_train, x_test, y_test, boost_mode='sklearn', show_progress=False):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in tqdm(range(num_factors), disable=(not show_progress)):
        if boost_mode == 'sklearn':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier()
        elif boost_mode == 'xgboost':
            from xgboost import XGBClassifier
            model = XGBClassifier()
        elif boost_mode == 'lightgbm':
            from lightgbm import LGBMClassifier
            model = LGBMClassifier()
        else:
            raise KeyError(f'Invalid boosting mode: {boost_mode=}')

        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))

    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def _disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11, base=importance_matrix.shape[1])


def _disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = _disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()
    return np.sum(per_code * code_importance)


def _completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11, base=importance_matrix.shape[0])


def _completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = _completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
