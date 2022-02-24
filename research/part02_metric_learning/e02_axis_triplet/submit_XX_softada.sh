#!/bin/bash


# OVERVIEW:
# - this experiment is designed to find the optimal triplet hyper-parameters

# OUTCOMES:
# - ???


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="TEST-softada-triplet"
export PARTITION="stampede"
export PARALLELISM=32

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours


# SWEEP FOR GOOD PARAMS
#   1 * (2*7*3) = 42
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='softada-sweep' \
    hydra.job.name="softada" \
    \
    run_length=short \
    metrics=fast \
    \
    settings.framework.beta=0.01 \
    framework=X--softadatvae \
    schedule=none \
    settings.model.z_size=25 \
    \
    framework.cfg.triplet_margin_max=NULL \
    framework.cfg.triplet_scale=1.0 \
    framework.cfg.triplet_p=1 \
    framework.cfg.detach_decoder=FALSE,TRUE \
    framework.cfg.triplet_loss=triplet_soft \
    \
    framework.cfg.ada_thresh_ratio=0.5,0.25,0.75 \
    framework.cfg.softada_scale_slope=0.75,0.5,0.25,0.0,-0.25,-0.5,-0.75 \
    \
    dataset=X--xysquares \
    sampling=gt_dist__manhat
