#!/bin/bash


# OVERVIEW:
# - this experiment is designed to find the optimal ada-triplet function and hyper-parameters

# OUTCOMES:
# - ???


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p02e03_unsupervised-axis-triplet"
export PARTITION="stampede"
export PARALLELISM=24

# the path to the generated arguments file
# - this needs to before we source the helper file
ARGS_FILE="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_01_$PROJECT.txt"

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Generate Experiment                                                       #
# ========================================================================= #

# SWEEP FOR GOOD UNSUPERVISED DO-ADA-TVAE PARAMS
#   1 * (2*4*2*5) = 80
ARGS_FILE="$ARGS_FILE" gen_sbatch_args_file \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_dotvae_hard_params_longmed' \
    hydra.job.name="dotvae_hparams_longmed" \
    \
    run_length=longmed \
    metrics=all \
    \
    settings.framework.beta=0.01 \
    settings.model.z_size=25 \
    \
    framework.cfg.triplet_loss=triplet \
    framework.cfg.overlap_num=512,1024 \
    sampling=gt_dist__manhat_scaled \
    framework.cfg.ada_thresh_mode=dist \
    \
    framework=X--dotvae \
    framework.cfg.detach_decoder=FALSE \
    \
    schedule=adanegtvae_up_ratio \
    framework.cfg.ada_thresh_ratio=0.5 \
    framework.cfg.adat_triplet_share_scale=0.5 \
    \
    framework.cfg.triplet_margin_max=10.0 \
    framework.cfg.triplet_scale=1.0 \
    framework.cfg.triplet_p=1 \
    \
    dataset=cars3d,smallnorb,shapes3d,dsprites \
    \
    settings.framework.recon_loss=mse \
    \
    framework.cfg.overlap_loss=NULL \
    framework.cfg.overlap_mine_ratio=0.1,0.2 \
    framework.cfg.overlap_mine_triplet_mode=none,hard_neg,semi_hard_neg,hard_pos,easy_pos \
    \
    framework.cfg.overlap_augment_mode=augment \
    framework.cfg.overlap_augment=NULL

# -- part of the above sweep!
# 1 * (2*1*2*5) = 20
ARGS_FILE="$ARGS_FILE" APPEND_ARGS=1 ARGS_START_NUM=81 gen_sbatch_args_file \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_dotvae_hard_params_longmed_xy' \
    hydra.job.name="dotvae_hparams_longmed_xy" \
    \
    run_length=longmed \
    metrics=all \
    \
    settings.framework.beta=0.01 \
    settings.model.z_size=25 \
    \
    framework.cfg.triplet_loss=triplet \
    framework.cfg.overlap_num=512,1024 \
    sampling=gt_dist__manhat_scaled \
    framework.cfg.ada_thresh_mode=dist \
    \
    framework=X--dotvae \
    framework.cfg.detach_decoder=FALSE \
    \
    schedule=adanegtvae_up_ratio \
    framework.cfg.ada_thresh_ratio=0.5 \
    framework.cfg.adat_triplet_share_scale=0.5 \
    \
    framework.cfg.triplet_margin_max=10.0 \
    framework.cfg.triplet_scale=1.0 \
    framework.cfg.triplet_p=1 \
    \
    dataset=X--xysquares \
    \
    +VAR.recon_loss_weight=1.0 \
    +VAR.kernel_loss_weight=3969.0 \
    +VAR.kernel_radius=31 \
    settings.framework.recon_loss='mse_box_r${VAR.kernel_radius}_l${VAR.recon_loss_weight}_k${VAR.kernel_loss_weight}' \
    \
    framework.cfg.overlap_loss=NULL \
    framework.cfg.overlap_mine_ratio=0.1,0.2 \
    framework.cfg.overlap_mine_triplet_mode=none,hard_neg,semi_hard_neg,hard_pos,easy_pos \
    \
    framework.cfg.overlap_augment_mode=augment \
    framework.cfg.overlap_augment=NULL

# ========================================================================= #
# Run Experiment                                                            #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

ARGS_FILE="$ARGS_FILE" submit_sbatch_args_file
