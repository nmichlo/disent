#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-axis-triplet-5.0"
export PARTITION="stampede"
export PARALLELISM=16

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

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
    system.framework.cfg_cls.triplet_loss=triplet \
    system.framework.cfg_cls.triplet_margin_min=0.001 \
    system.framework.cfg_cls.triplet_margin_max=1 \
    system.framework.cfg_cls.triplet_scale=0.1 \
    system.framework.cfg_cls.triplet_p=1 \
    \
    system.framework.cfg_cls.detach=FALSE \
    system.framework.cfg_cls.detach_decoder=FALSE \
    system.framework.cfg_cls.detach_no_kl=FALSE \
    system.framework.cfg_cls.detach_std=NULL \
    \
    system.framework.cfg_cls.ada_average_mode=gvae \
    system.framework.cfg_cls.ada_thresh_mode=dist \
    system.framework.cfg_cls.ada_thresh_ratio=0.5 \
    \
    system.framework.cfg_cls.adat_triplet_ratio=1.0 \
    system.framework.cfg_cls.adat_triplet_pull_weight=-1.0,-0.1,0.0,0.1,1.0 \
    \
    system.framework.cfg_cls.adat_share_mask_mode=posterior
