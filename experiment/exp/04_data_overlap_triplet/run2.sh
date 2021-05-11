#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-data-overlap-triplet"
export PARTITION="batch"
export PARALLELISM=16

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# 1 * (2*8*4) == 64
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='best-augment-strength__alt' \
    \
    framework=X--dotvae_aug \
    run_length=short \
    model=conv64alt \
    model.z_size=25 \
    \
    specializations.data_wrapper='gt_dist_${framework.data_wrap_mode}' \
    schedule=adavae_up_ratio_full,adavae_up_all_full \
    sampling=gt_dist_manhat \
    sampling.triplet_swap_chance=0 \
    dataset=xysquares \
    \
    framework.module.triplet_loss=triplet \
    framework.module.triplet_margin_min=0.001 \
    framework.module.triplet_margin_max=1 \
    framework.module.triplet_scale=0.1 \
    framework.module.triplet_p=1 \
    \
    framework.module.detach=FALSE \
    framework.module.detach_decoder=FALSE \
    framework.module.detach_no_kl=FALSE \
    framework.module.detach_logvar=NULL \
    \
    framework.module.ada_average_mode=gvae \
    framework.module.ada_thresh_mode=dist \
    framework.module.ada_thresh_ratio=0.5 \
    \
    framework.module.adat_triplet_share_scale=1.0 \
    \
    framework.module.adat_share_mask_mode=posterior \
    \
    framework.module.overlap_augment_mode='augment' \
    framework.module.overlap_augment.kernel=xy1_r47,xy8_r47,box_r47,gau_r47 \
    \
    framework.module.overlap_num=4096 \
    framework.module.overlap_mine_ratio=0.1 \
    framework.module.overlap_mine_triplet_mode=none,hard_neg,semi_hard_neg,hard_pos,easy_pos,ran:hard_neg+hard_pos,ran:hard_neg+easy_pos,ran:hard_pos+easy_pos

  # framework.module.overlap_augment.kernel=xy1_r47,xy8_r47,box_r47,gau_r47,box_r15,box_r31,box_r63,gau_r15,gau_r31,gau_r63
