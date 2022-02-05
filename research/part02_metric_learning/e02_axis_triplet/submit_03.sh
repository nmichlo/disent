#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-03__axis-triplet-3.0"
export PARTITION="stampede"
export PARALLELISM=32

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# TODO: update this script
echo UPDATE THIS SCRIPT
exit 1

# LONG RUNS:
# - test best losses & best hparams from test1 on different datasets with long runs
#   + [not tested] triplet_soft_neg_ave
#   + triplet_all_soft_ave
#   + triplet_hard_neg_ave
#   + triplet_hard_neg_ave_pull

# 1 * (2*3*4*4=96) = 96
#submit_sweep \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='long-run' \
#    \
#    framework=X--adatvae \
#    run_length=long \
#    model.z_size=25 \
#    \
#    framework.cfg.triplet_margin_max=1.0 \
#    framework.cfg.triplet_scale=0.1 \
#    framework.cfg.triplet_p=1 \
#    sampling=gt_dist_manhat,gt_dist_manhat_scaled \
#    \
#    framework.cfg.thresh_ratio=0.5 \
#    framework.cfg.ada_triplet_ratio=1.0 \
#    framework.cfg.ada_triplet_soft_scale=1.0 \
#    framework.cfg.ada_triplet_sample=FALSE \
#    \
#    schedule=adavae_all,adavae_thresh,adavae_ratio \
#    framework.cfg.ada_triplet_loss=triplet,triplet_all_soft_ave,triplet_hard_neg_ave,triplet_hard_neg_ave_pull \
#    dataset=xysquares,shapes3d,cars3d,dsprites

# 2*2*3*4*4
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='med-run+datasets+swap-chance+manhat-scaled' \
    \
    framework=X--adatvae \
    run_length=medium \
    model.z_size=25 \
    \
    sampling=gt_dist_manhat_scaled,gt_dist_manhat \
    schedule=adavae_all,adavae_thresh,adavae_ratio \
    sampling.triplet_swap_chance=0,0.1 \
    \
    framework.cfg.triplet_margin_max=1.0 \
    framework.cfg.triplet_scale=0.1 \
    framework.cfg.triplet_p=1 \
    \
    framework.cfg.thresh_ratio=0.5 \
    framework.cfg.ada_triplet_ratio=1.0 \
    framework.cfg.ada_triplet_soft_scale=1.0 \
    framework.cfg.ada_triplet_sample=FALSE \
    \
    framework.cfg.ada_triplet_loss=triplet,triplet_all_soft_ave,triplet_hard_neg_ave,triplet_hard_neg_ave_pull \
    dataset=xysquares,shapes3d,cars3d,dsprites
