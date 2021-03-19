#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-naive-triplet"
export PARTITION="batch"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

# 3*2*2*3*2 = 72
submit_sweep \
    framework=tvae \
    dataset=xysquares \
    specializations.data_wrapper='gt_dist_${framework.data_wrap_mode}' \
    \
    sampling=gt_dist_factors,gt_dist_manhat,gt_dist_combined \
    framework.module.triplet_margin_min=0.001,0.1 \
    framework.module.triplet_margin_max=1.0,10.0 \
    framework.module.triplet_scale=1.0,0.1,0.01 \
    framework.module.triplet_p=1,2
