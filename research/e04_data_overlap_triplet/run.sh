##!/bin/bash
#
## ========================================================================= #
## Settings                                                                  #
## ========================================================================= #
#
#export PROJECT="exp-data-overlap-triplet"
#export PARTITION="stampede"
#export PARALLELISM=32
#
## source the helper file
#source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"
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
#    system.framework.cfg_cls.triplet_loss=triplet \
#    system.framework.cfg_cls.triplet_margin_min=0.001 \
#    system.framework.cfg_cls.triplet_margin_max=1 \
#    system.framework.cfg_cls.triplet_scale=0.1,0.01 \
#    system.framework.cfg_cls.triplet_p=1 \
#    \
#    system.framework.cfg_cls.detach=FALSE \
#    system.framework.cfg_cls.disable_decoder=FALSE \
#    system.framework.cfg_cls.detach_no_kl=FALSE \
#    system.framework.cfg_cls.detach_std=NULL \
#    \
#    system.framework.cfg_cls.ada_average_mode=gvae \
#    system.framework.cfg_cls.ada_thresh_mode=dist \
#    system.framework.cfg_cls.ada_thresh_ratio=0.5 \
#    \
#    system.framework.cfg_cls.adat_triplet_share_scale=0.95 \
#    \
#    system.framework.cfg_cls.adat_share_mask_mode=posterior \
#    \
#    system.framework.cfg_cls.overlap_num=4096 \
#    system.framework.cfg_cls.overlap_mine_ratio=0.05,0.1 \
#    system.framework.cfg_cls.overlap_mine_triplet_mode=none,hard_neg,semi_hard_neg,hard_pos,easy_pos \
#    \
#    system.framework.cfg_cls.overlap_augment_mode='augment' \
#    system.framework.cfg_cls.overlap_augment.p=1.0 \
#    system.framework.cfg_cls.overlap_augment.radius=[61,61],[0,61] \
#    system.framework.cfg_cls.overlap_augment.random_mode='batch' \
#    system.framework.cfg_cls.overlap_augment.random_same_xy=TRUE
