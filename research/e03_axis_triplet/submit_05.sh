#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-03__axis-triplet-5.0"
export PARTITION="stampede"
export PARALLELISM=16

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# TODO: update this script
echo UPDATE THIS SCRIPT
exit 1

# 1 * (3*6*5) == 90
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='ada-best-pull-weight' \
    \
    framework=X--adanegtvae \
    run_length=short,medium,long \
    model.z_size=25 \
    \
    schedule=adavae_down_all,adavae_up_all,adavae_down_ratio,adavae_up_ratio,adavae_down_thresh,adavae_up_thresh \
    sampling=gt_dist_manhat \
    sampling.triplet_swap_chance=0 \
    dataset=xysquares \
    \
    framework.cfg.triplet_loss=triplet \
    framework.cfg.triplet_margin_min=0.001 \
    framework.cfg.triplet_margin_max=1 \
    framework.cfg.triplet_scale=0.1 \
    framework.cfg.triplet_p=1 \
    \
    framework.cfg.detach=FALSE \
    framework.cfg.detach_decoder=FALSE \
    framework.cfg.detach_no_kl=FALSE \
    framework.cfg.detach_std=NULL \
    \
    framework.cfg.ada_average_mode=gvae \
    framework.cfg.ada_thresh_mode=dist \
    framework.cfg.ada_thresh_ratio=0.5 \
    \
    framework.cfg.adat_triplet_ratio=1.0 \
    framework.cfg.adat_triplet_pull_weight=-1.0,-0.1,0.0,0.1,1.0 \
    \
    framework.cfg.adat_share_mask_mode=posterior
