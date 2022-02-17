#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-03__axis-triplet-4.0"
export PARTITION="stampede"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# TODO: update this script
echo UPDATE THIS SCRIPT
exit 1

# RESULT:
# - BAD: ada_thresh_mode=symmetric_kl, rather use "dist"
# - BAD: framework.cfg.adaave_decode_orig=FALSE, rather use TRUE
# - adat_share_ave_mode depends on other settings, but usually doesnt matter
# - adaave_augment_orig depends on other settings, but usually doesnt matter
# - GOOD: adat_triplet_loss=triplet_hard_neg_ave
# - NOTE: schedule=adavae_up_ratio  usually converges sooner
# - NOTE: schedule=adavae_up_all    usually converges later (makes sense because its a doubling effect a ^ 2)
# - NOTE: schedule=adavae_up_thresh usually is worse at converging


# 3*2*4*2*2*2 == 192
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='short-run__ada-best-loss-combo' \
    \
    framework=X--adaavetvae \
    run_length=short \
    model.z_size=25 \
    \
    schedule=adavae_up_all,adavae_up_ratio,adavae_up_thresh \
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
    framework.module.ada_average_mode=gvae \
    framework.module.ada_thresh_mode=symmetric_kl,dist \
    framework.module.ada_thresh_ratio=0.5 \
    \
    framework.module.adat_triplet_loss=triplet,triplet_soft_ave_all,triplet_hard_neg_ave,triplet_hard_ave_all \
    framework.module.adat_triplet_ratio=1.0 \
    framework.module.adat_triplet_soft_scale=1.0 \
    framework.module.adat_triplet_pull_weight=0.1 \
    \
    framework.module.adat_share_mask_mode=posterior \
    framework.module.adat_share_ave_mode=all,neg \
    \
    framework.module.adaave_augment_orig=TRUE,FALSE \
    framework.module.adaave_decode_orig=TRUE,FALSE

# TRY THESE TOO:
# framework.module.adat_share_ave_mode=all,neg,pos,pos_neg \
# framework.module.adat_share_mask_mode=posterior,sample,sample_each \
# framework.module.adat_triplet_loss=triplet,triplet_soft_ave_all,triplet_hard_neg_ave,triplet_hard_neg_ave_pull,triplet_hard_ave_all \

# # 3*2*8*2*3*2*2
#submit_sweep \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='short-run__ada-best-loss-combo' \
#    \
#    framework=X--adaavetvae \
#    run_length=short \
#    model.z_size=25 \
#    \
#    schedule=adavae_all,adavae_thresh,adavae_ratio \
#    sampling=gt_dist_manhat \
#    sampling.triplet_swap_chance=0 \
#    dataset=xysquares \
#    \
#    triplet_loss=triplet \
#    triplet_margin_min=0.001 \
#    triplet_margin_max=1 \
#    triplet_scale=0.1 \
#    triplet_p=1 \
#    \
#    detach=FALSE \
#    detach_decoder=FALSE \
#    detach_no_kl=FALSE \
#    detach_std=NULL \
#    \
#    ada_average_mode=gvae \
#    ada_thresh_mode=symmetric_kl,dist \
#    ada_thresh_ratio=0.5 \
#    \
#    adat_triplet_loss=triplet,triplet_soft_ave_neg,triplet_soft_ave_p_n,triplet_soft_ave_all,triplet_hard_ave,triplet_hard_neg_ave,triplet_hard_neg_ave_pull,triplet_hard_ave_all \
#    adat_triplet_ratio=1.0 \
#    adat_triplet_soft_scale=1.0 \
#    adat_triplet_pull_weight=0.1 \
#    \
#    adat_share_mask_mode=posterior,dist \
#    adat_share_ave_mode=all,pos_neg,pos,neg \
#    \
#    adaave_augment_orig=TRUE,FALSE \
#    adaave_decode_orig=TRUE,FALSE
