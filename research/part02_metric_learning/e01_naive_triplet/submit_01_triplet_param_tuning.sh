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
#   1 * (1*2*2*1*2*5*2) = 160
#   * triplet_scale: 0.01 is too low (from provisional experiments)
#   * triplet_margin_max: 0.1 is too low (from provisional experiments)
#   * beta: 0.01 (and 0.0316) chosen from previous experiment sweeps
# results:
#   - z_size: generally worse disentanglement on larger z, so lets make the problem harder and use 25 later for ada methods.
#   - triplet_margin_max: 10 seems good (maybe too strong), 1 is ok (maybe too weak), 0.1 is too weak
#   - triplet_scale: 1.0 is good, 0.1 is ok but too weak often, 0.01 is definitely too weak.
#   - triplet_p: 2 somtimes produces better results, but generally 1 is better for the linearity ratio and latent-ground correspondance.
#   - dataset: actually struggles on cars3d, xysquares needs strong triplet to learn.
#   - sampling: both `gt_dist__manhat` and `gt_dist__manhat_scaled` seem ok, need to test with ADA methods. Maybe `gt_dist__manhat` for simplicity?
# summary:
#   - triplet_margin_max=10.0, triplet_scale=1.0, triplet_p=1, sampling=gt_dist__manhat,gt_dist__manhat_scaled, z_size=25
# todo:
#   - detach_decoder: see how much this affects things
#   - triplet_loss: see if other versions are better
# MAKE PLOTS:
# -- use this to figure out if l1 or l2 is better
# -- use this to hparam tune for later experiments
# SWEEP FOR GOOD PARAMS - RERUN:
# changes:
# - no longer checking z_size=9,25
# + added detach_decoder=FALSE,TRUE
# results:
# - pretty much the same as before, just needed the fixed metrics...
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_tvae_params_basic_RERUN' \
    hydra.job.name="tvae_params" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.01 \
    framework=tvae \
    schedule=none \
    settings.model.z_size=25 \
    \
    framework.cfg.triplet_margin_max=1.0,10.0 \
    framework.cfg.triplet_scale=0.1,1.0 \
    framework.cfg.triplet_p=1,2 \
    framework.cfg.detach_decoder=FALSE,TRUE \
    framework.cfg.triplet_loss=triplet \
    \
    dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares \
    sampling=gt_dist__manhat,gt_dist__manhat_scaled

submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_tvae_params_basic_RERUN_soft' \
    hydra.job.name="tvae_params" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.01 \
    framework=tvae \
    schedule=none \
    settings.model.z_size=25 \
    \
    framework.cfg.triplet_margin_max=1.0,10.0 \
    framework.cfg.triplet_scale=0.1,1.0 \
    framework.cfg.triplet_p=1,2 \
    framework.cfg.detach_decoder=FALSE,TRUE \
    framework.cfg.triplet_loss=triplet_soft \
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
# RERUN NOTES:
# -- we no longer want this because triplet_soft doesnt seem to work
#    well with axis-aligned triplet or detached decoders...
# 1 * (2*2*2*2*5) = 80
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
#    sampling=gt_dist__manhat,gt_dist__manhat_scaled
#    \
#    framework.cfg.detach_decoder=FALSE,TRUE \
#    framework.cfg.triplet_margin_max=10.0 \
#    framework.cfg.triplet_scale=1.0 \
#    framework.cfg.triplet_p=1,2 \
#    framework.cfg.triplet_loss=triplet_soft,triplet \
#    \
#    dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares
