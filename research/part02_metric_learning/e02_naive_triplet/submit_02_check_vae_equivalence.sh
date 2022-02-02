#!/bin/bash

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
