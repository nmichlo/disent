#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-axis-triplet-3.0"
export PARTITION="batch"
export PARALLELISM=30

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# 2 * (3*4*4=48) = 96
submit_sweep \
    +DUMMY.repeat=1,2 \
    +EXTRA.tags='long-run' \
    \
    framework=X--adatvae \
    run_length=long \
    model.z_size=25 \
    \
    framework.module.triplet_margin_max=1.0 \
    framework.module.triplet_scale=0.1 \
    framework.module.triplet_p=1 \
    sampling=gt_dist_manhat \
    specializations.data_wrapper='gt_dist_${framework.data_wrap_mode}' \
    \
    framework.module.thresh_ratio=0.5 \
    framework.module.ada_triplet_ratio=1.0 \
    framework.module.ada_triplet_soft_scale=1.0 \
    framework.module.ada_triplet_sample=FALSE \
    \
    schedule=adavae_all,adavae_thresh,adavae_ratio \
    framework.module.ada_triplet_loss=triplet,triplet_all_soft_ave,triplet_hard_neg_ave,triplet_hard_neg_ave_pull \
    dataset=xysquares,shapes3d,cars3d,dsprites
