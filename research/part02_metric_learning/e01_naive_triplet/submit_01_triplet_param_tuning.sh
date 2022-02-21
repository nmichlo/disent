#!/bin/bash


# OVERVIEW:
# - this experiment is designed to find the optimal triplet hyper-parameters

# OUTCOMES:
# - ???


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p02e01_triplet-param-tuning"
export PARTITION="stampede"
export PARALLELISM=32

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours


# SWEEP FOR GOOD TVAE PARAMS
#   1 * (1*2*2*2*2*5*2) = 160
#   * triplet_scale: 0.01 is too low (from provisional experiments)
#   * triplet_margin_max: 0.1 is too low (from provisional experiments)
#   * beta: 0.01 (and 0.0316) chosen from previous experiment sweeps
# results:
#   - z_size:
#   - triplet_margin_max:
#   - triplet_scale:
#   - triplet_p:
#   - dataset:
#   - sampling:
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_tvae_params_basic' \
    hydra.job.name="tvae_params" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.01 \
    framework=tvae \
    schedule=none \
    settings.model.z_size=9,25 \
    \
    framework.cfg.triplet_margin_max=1.0,10.0 \
    framework.cfg.triplet_scale=0.1,1.0 \
    framework.cfg.triplet_p=1,2 \
    \
    dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares \
    sampling=gt_dist__manhat,gt_dist__manhat_scaled

# TRIPLET TYPES:
# - N/A:     [triplet_soft]
# - max:     [triplet, triplet_sigmoid]
# - max+min: [min_clamped_triplet, split_clamped_triplet]

# TRIPLET OPTIONS:
# framework.cfg.detach_decoder: FALSE
# framework.cfg.triplet_loss: triplet
# framework.cfg.triplet_margin_min: 0.001
# framework.cfg.triplet_margin_max: 1
# framework.cfg.triplet_scale: 0.1
# framework.cfg.triplet_p: 1

# SWEEP TRIPLET MODES:
# 1 * (2*1*2*3*5*2) = 48
#submit_sweep \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='sweep_tvae_modes_basic' \
#    hydra.job.name="tvae_modes" \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    settings.framework.beta=0.01 \
#    framework=tvae \
#    schedule=none \
#    settings.model.z_size=25 \
#    \
#    framework.cfg.triplet_margin_max=1.0,10.0 \
#    framework.cfg.triplet_scale=1.0 \
#    framework.cfg.triplet_p=1 \
#    framework.cfg.detach_decoder=TRUE,FALSE \
#    framework.cfg.triplet_loss=triplet,triplet_sigmoid,triplet_soft \
#    \
#    dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares \
#    sampling=gt_dist__manhat,gt_dist__manhat_scaled
