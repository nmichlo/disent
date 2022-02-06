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
export PROJECT="final-02__naive-triplet-hparams"
export PARTITION="batch"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

# general sweep of hyper parameters for triplet
# 1 * (3*3*3*2*3 = 162) = 162
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_tvae_params' \
    \
    run_length=long \
    metrics=all \
    \
    framework=tvae \
    settings.framework.beta=0.0316,0.01,0.1 \
    \
    framework.cfg.triplet_margin_max=0.1,1.0,10.0 \
    framework.cfg.triplet_scale=0.1,1.0,0.01 \
    framework.cfg.triplet_p=1,2 \
    \
    dataset=xysquares,cars3d,smallnorb \
    sampling=gt_dist__manhat

# check sampling strategy
# 2 * (4 * 5 = 20) = 40
echo PARAMS NOT SET FROM PREVIOUS SWEEP
exit 1

# TODO: set the parameters
submit_sweep \
    +DUMMY.repeat=1,2 \
    +EXTRA.tags='sweep_tvae_sampling' \
    \
    run_length=long \
    metrics=all \
    \
    framework=tvae \
    settings.framework.beta=??? \
    \
    framework.cfg.triplet_margin_max=??? \
    framework.cfg.triplet_scale=??? \
    framework.cfg.triplet_p=??? \
    \
    dataset=xysquares,cars3d,shapes3d,dsprites,smallnorb \
    sampling=gt_dist__manhat_scaled,gt_dist__manhat,gt__dist_combined,gt_dist__factors
