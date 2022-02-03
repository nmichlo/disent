#!/bin/bash

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
