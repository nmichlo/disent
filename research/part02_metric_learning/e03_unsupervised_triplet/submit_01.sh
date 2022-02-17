##!/bin/bash
#
## ========================================================================= #
## Settings                                                                  #
## ========================================================================= #
#
#export USERNAME="n_michlo"
#export PROJECT="final-04__data-overlap-triplet"
#export PARTITION="stampede"
#export PARALLELISM=32
#
## source the helper file
#source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"
#
## ========================================================================= #
## Experiment                                                                #
## ========================================================================= #
#
#clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours
#
## 1 * (3*2*2*5*2) == 120
#submit_sweep \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='med-best' \
#    \
#    framework=X--dotvae_aug \
#    run_length=medium \
#    model.z_size=25 \
#    \
#    schedule=adavae_up_all,adavae_up_ratio,none \
#    sampling=gt_dist_manhat \
#    sampling.triplet_swap_chance=0 \
#    dataset=xysquares \
#    \
#    framework.cfg.triplet_loss=triplet \
#    framework.cfg.triplet_margin_min=0.001 \
#    framework.cfg.triplet_margin_max=1 \
#    framework.cfg.triplet_scale=0.1,0.01 \
#    framework.cfg.triplet_p=1 \
#    \
#    framework.cfg.detach=FALSE \
#    framework.cfg.detach_decoder=FALSE \
#    framework.cfg.detach_no_kl=FALSE \
#    framework.cfg.detach_std=NULL \
#    \
#    framework.cfg.ada_average_mode=gvae \
#    framework.cfg.ada_thresh_mode=dist \
#    framework.cfg.ada_thresh_ratio=0.5 \
#    \
#    framework.cfg.adat_triplet_share_scale=0.95 \
#    \
#    framework.cfg.adat_share_mask_mode=posterior \
#    \
#    framework.cfg.overlap_num=4096 \
#    framework.cfg.overlap_mine_ratio=0.05,0.1 \
#    framework.cfg.overlap_mine_triplet_mode=none,hard_neg,semi_hard_neg,hard_pos,easy_pos \
#    \
#    framework.cfg.overlap_augment_mode='augment' \
#    framework.cfg.overlap_augment.p=1.0 \
#    framework.cfg.overlap_augment.radius=[61,61],[0,61] \
#    framework.cfg.overlap_augment.random_mode='batch' \
#    framework.cfg.overlap_augment.random_same_xy=TRUE
