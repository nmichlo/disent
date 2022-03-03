#!/bin/bash


# OVERVIEW:
# - this experiment is designed to find the optimal ada-triplet function and hyper-parameters

# OUTCOMES:
# - ???


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p02e02_axis-aligned-triplet"
export PARTITION="stampede"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 172800 "C-disent" # 48 hours

# SWEEP FOR GOOD TVAE PARAMS
#   1 * (3*2*5*2*5) = 135
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_adanegtvae_params_basic' \
    hydra.job.name="adanegtvae_params" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.01 \
    settings.model.z_size=25 \
    settings.optimizer.lr=1e-3,3e-4,1e-4 \
    \
    framework.cfg.detach_decoder=FALSE \
    \
    sampling=gt_dist__manhat,gt_dist__manhat_scaled \
    schedule=adavae_up_all,adavae_up_all_full,adavae_up_ratio,adavae_up_ratio_full,adavae_up_thresh \
    \
    framework=X--adanegtvae \
    framework.cfg.ada_thresh_mode=symmetric_kl,dist \
    \
    framework.cfg.triplet_margin_max=10.0 \
    framework.cfg.triplet_scale=1.0 \
    framework.cfg.triplet_p=1 \
    framework.cfg.triplet_loss=triplet_soft \
    \
    dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares


# CHECK ADAVAE modes:
# framework.cfg.adat_triplet_loss=triplet,triplet_soft_ave_neg,triplet_soft_ave_p_n,triplet_soft_ave_all,triplet_hard_ave,triplet_hard_neg_ave,triplet_hard_neg_ave_pull,triplet_hard_ave_all,triplet_hard_neg_ave_scaled \

# TODO:
# - try run length long
# - try run detach
