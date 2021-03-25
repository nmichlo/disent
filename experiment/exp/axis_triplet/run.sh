#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-axis-triplet-3.0"
export PARTITION="batch"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes batch 28800 "C-disent" # 8 hours

# 1 * 2 * (3*2*6=36) = 72
submit_sweep \
    +DUMMY.repeat=1,2 \
    \
    framework=X--adatvae \
    dataset=xysquares \
    run_length=short \
    \
    framework.module.triplet_margin_max=1.0 \
    framework.module.triplet_scale=0.1 \
    framework.module.triplet_p=1 \
    sampling=gt_dist_manhat \
    specializations.data_wrapper='gt_dist_${framework.data_wrap_mode}' \
    \
    model.z_size=25,9 \
    \
    framework.module.thresh_ratio=0.5
    framework.module.ada_triplet_ratio=1.0
    schedule=none,adavae_ratio,adavae_thresh
    framework.module.ada_triplet_sample=TRUE,FALSE
    framework.module.ada_triplet_loss=triplet_soft_ave,triplet_soft_neg_ave,triplet_all_soft_ave,triplet_hard_ave,triplet_hard_neg_ave,triplet_all_hard_ave
