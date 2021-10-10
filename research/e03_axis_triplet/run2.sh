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

# MED RUNS:
# - test for best hparams for all soft ave loss
# 2 * (2*3*3*3=54) = 104
submit_sweep \
    +DUMMY.repeat=1,2 \
    +EXTRA.tags='med-run+soft-hparams' \
    \
    framework=X--adatvae \
    run_length=medium \
    model.z_size=25 \
    \
    framework.module.triplet_margin_max=1.0,5.0 \
    framework.module.triplet_scale=0.1,0.02,0.5 \
    framework.module.triplet_p=1 \
    sampling=gt_dist_manhat \
    \
    framework.module.thresh_ratio=0.5 \
    framework.module.ada_triplet_ratio=1.0 \
    framework.module.ada_triplet_soft_scale=0.25,1.0,4.0 \
    framework.module.ada_triplet_sample=FALSE \
    \
    schedule=adavae_all,adavae_thresh,adavae_ratio \
    framework.module.ada_triplet_loss=triplet_all_soft_ave \
    dataset=xysquares
