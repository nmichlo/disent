#!/bin/bash


# OVERVIEW:
# - this experiment is designed to find the optimal ada-triplet function and hyper-parameters

# OUTCOMES:
# - ???


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p02e02_axis-aligned-triplet"
export PARTITION="stampede"
export PARALLELISM=32

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# TODO: design experiment

# SWEEP FOR GOOD TVAE PARAMS
#   1 * (5*9*3*1) = 135
#submit_sweep \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='sweep_tvae_params_basic' \
#    hydra.job.name="tvae_params" \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    settings.framework.beta=0.01 \
#    framework=tvae \
#    settings.model.z_size=25 \
#    \
#    framework.cfg.triplet_scale=1.0 \
#    framework.cfg.triplet_p=1 \
#    framework.cfg.detach_decoder=FALSE \
#    framework.cfg.triplet_loss=triplet_soft \
#    \
#    schedule=adavae_up_all,adavae_up_all_full,adavae_up_ratio,adavae_up_ratio_full,adavae_up_thresh \
#    framework.cfg.adat_triplet_loss=triplet,triplet_soft_ave_neg,triplet_soft_ave_p_n,triplet_soft_ave_all,triplet_hard_ave,triplet_hard_neg_ave,triplet_hard_neg_ave_pull,triplet_hard_ave_all,triplet_hard_neg_ave_scaled \
#    \
#    dataset=smallnorb,shapes3d,X--xysquares \
#    sampling=gt_dist__manhat
#
