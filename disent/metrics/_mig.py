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
Mutual Information Gap from the beta-TC-VAE paper.
Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""

import logging

import numpy as np

from disent.dataset import DisentDataset
from disent.metrics import utils


log = logging.getLogger(__name__)


# ========================================================================= #
# Mutual Information Gap                                                    #
# ========================================================================= #


def metric_mig(
        dataset: DisentDataset,
        representation_function,
        num_train=10000,
        batch_size=16,
):
    """Computes the mutual information gap.
    Args:
      dataset: DisentDataset to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      num_train: Number of points used for training.
      batch_size: Batch size for sampling.
    Returns:
      Dict with average mutual information gap.
    """
    log.debug("Generating training set.")
    mus_train, ys_train = utils.generate_batch_factor_code(dataset, representation_function, num_train, batch_size)
    assert mus_train.shape[1] == num_train
    return _compute_mig(mus_train, ys_train)


def _compute_mig(mus_train, ys_train):
    """
    Computes score based on both training and testing codes and factors.
    """
    discretized_mus = utils.histogram_discretize(mus_train, num_bins=20)
    m = utils.discrete_mutual_info(discretized_mus, ys_train)
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    # m is [num_latents, num_factors]
    entropy = utils.discrete_entropy(ys_train)
    sorted_m = np.sort(m, axis=0)[::-1]
    return {
        "mig.discrete_score": np.mean(np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    }
