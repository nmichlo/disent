#!/bin/bash

#
# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# MIT License
#
# Copyright (c) 2022 Nathan Juraj Michlo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-02__naive-triplet-equivalence"
export PARTITION="batch"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

# make sure the tvae is actually working
# like a vae when the triplet loss is disabled
# 1 * (4=4) = 4
submit_sweep \
    +DUMMY.repeat=1,2 \
    +EXTRA.tags='check_equivalence' \
    \
    run_length=medium \
    metrics=all \
    \
    framework=tvae \
    framework.cfg.triplet_scale=0.0 \
    settings.framework.beta=0.0316 \
    \
    dataset=xysquares \
    sampling=gt_dist__manhat_scaled,gt_dist__manhat,gt__dist_combined,gt_dist__factors

# check how sampling effects beta and adavae
# 2 * (2*3=6) = 12
submit_sweep \
    +DUMMY.repeat=1,2 \
    +EXTRA.tags='check_vae_sampling' \
    \
    run_length=medium \
    metrics=all \
    \
    framework=betavae,adavae \
    settings.framework.beta=0.0316 \
    \
    dataset=xysquares \
    sampling=gt_dist__manhat_scaled,gt_dist__manhat,gt__dist_combined,gt_dist__factors
