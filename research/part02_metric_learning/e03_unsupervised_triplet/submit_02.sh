#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-04__data-overlap-triplet"
export PARTITION="batch"
export PARALLELISM=16

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# TODO: update this script
echo UPDATE THIS SCRIPT
exit 1

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
    schedule=adavae_up_ratio_full,adavae_up_all_full \
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
    framework.cfg.adat_triplet_share_scale=1.0 \
    \
    framework.cfg.adat_share_mask_mode=posterior \
    \
    framework.cfg.overlap_augment_mode='augment' \
    framework.cfg.overlap_augment.kernel=xy1_r47,xy8_r47,box_r47,gau_r47 \
    \
    framework.cfg.overlap_num=4096 \
    framework.module.overlap_mine_ratio=0.1 \
    framework.module.overlap_mine_triplet_mode=none,hard_neg,semi_hard_neg,hard_pos,easy_pos,ran:hard_neg+hard_pos,ran:hard_neg+easy_pos,ran:hard_pos+easy_pos

  # framework.module.overlap_augment.kernel=xy1_r47,xy8_r47,box_r47,gau_r47,box_r15,box_r31,box_r63,gau_r15,gau_r31,gau_r63
