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

# SWEEP FOR GOOD ADANEG-TVAE PARAMS (OLD)
#   1 * (2*2*5*2*5) = 200
# OUTCOMES:
# - this experiment was wrong! The schedules were not set up correctly! we did not define
#   the correct initial values for `adat_triplet_share_scale` or `ada_thresh_ratio`
# - HOWEVER: The learning rate less than 1e-3 did not really work well, the batch size of
#            256 is large enough that we can use this high learning rate.
# - HOWEVER: On average: ada_thresh_mode="dist" performs FAR better than "symmetric_kl"
# - HOWEVER: On average, gt_dist__manhat_scaled sometimes performs better, but metrics and distance measurement don't line up properly.
#submit_sweep \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='sweep_adanegtvae_params_basic' \
#    hydra.job.name="adanegtvae_params" \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    settings.framework.beta=0.01 \
#    settings.model.z_size=25 \
#    settings.optimizer.lr=1e-3,3e-4 \
#    \
#    framework.cfg.detach_decoder=FALSE \
#    \
#    sampling=gt_dist__manhat,gt_dist__manhat_scaled \
#    schedule=OLD_adavae_up_all,OLD_adavae_up_all_full,OLD_adavae_up_ratio,OLD_adavae_up_ratio_full,OLD_adavae_up_thresh \
#    \
#    framework=X--adanegtvae \
#    framework.cfg.ada_thresh_mode=symmetric_kl,dist \
#    \
#    framework.cfg.triplet_margin_max=10.0 \
#    framework.cfg.triplet_scale=1.0 \
#    framework.cfg.triplet_p=1 \
#    framework.cfg.triplet_loss=triplet_soft \
#    \
#    dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares

# Try These:
# - adavae modes: framework.cfg.adat_triplet_loss=triplet,triplet_soft_ave_neg,triplet_soft_ave_p_n,triplet_soft_ave_all,triplet_hard_ave,triplet_hard_neg_ave,triplet_hard_neg_ave_pull,triplet_hard_ave_all,triplet_hard_neg_ave_scaled \
# - try run length long
# - try run detach


# SWEEP FOR GOOD ADANEG-TVAE PARAMS
#   1 * (2*5*2*5) = 100
# OUTCOMES:
# - `gt_dist__manhat_scaled` is generally better (problem is metrics sometimes need to be scaled to match this, so metrics might actaully be off?)
# - `adanegtvae_up_thresh` is generally bad, does not converge well. Might converge better for longer runs if givem more time? Usually quite stable though, recon loss is decent.
# - `*_full` version of schedules generally converge a bit quicker, but the recon loss decays badly towards the end.
# - `!*_full` versions (the not full versions) might get better scores if given more time, are generally more stable too in terms of recon loss.
# - `dist` is MUCH better than `symmetric_kl` --- always use the former!
# NOTE:
# -- distances don't always correlate well, especially if the dataset is more difficult to learn in terms of recon loss.
#    we could try and increase the triplet_scale to compensate for this in future experiments?
#    OR: we can detach the decoder?
# -- I am not sure if the soft-margin formulation hurts learning of
#    distances? It might, try revert to normal triplet loss?
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_adanegtvae_params_longmed' \
    hydra.job.name="adanegtvae_hparams_longmed" \
    \
    run_length=longmed \
    metrics=all \
    \
    settings.framework.beta=0.01 \
    settings.model.z_size=25 \
    \
    sampling=gt_dist__manhat,gt_dist__manhat_scaled \
    framework.cfg.ada_thresh_mode=dist,symmetric_kl \
    \
    framework=X--adanegtvae \
    framework.cfg.detach_decoder=FALSE \
    \
    schedule=adanegtvae_up_all,adanegtvae_up_all_full,adanegtvae_up_ratio,adanegtvae_up_ratio_full,adanegtvae_up_thresh \
    framework.cfg.ada_thresh_ratio=0.5 \
    framework.cfg.adat_triplet_share_scale=0.5 \
    \
    framework.cfg.triplet_margin_max=10.0 \
    framework.cfg.triplet_scale=1.0 \
    framework.cfg.triplet_p=1 \
    framework.cfg.triplet_loss=triplet_soft \
    \
    dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares


# SWEEP FOR ALTERNATIVE ADANEG-TVAE PARAMS
#   1 * (2*2*2*2*5) = 80
# OUTCOMES:
# - ???
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_adanegtvae_alt_params_longmed' \
    hydra.job.name="adanegtvae_hparams_alt" \
    \
    run_length=longmed \
    metrics=all \
    \
    settings.framework.beta=0.01 \
    settings.model.z_size=25 \
    \
    sampling=gt_dist__manhat_scaled \
    framework.cfg.ada_thresh_mode=dist \
    \
    schedule=adanegtvae_up_ratio,adanegtvae_up_all \
    framework.cfg.triplet_scale=10.0,1.0 \
    framework.cfg.detach_decoder=FALSE,TRUE \
    framework.cfg.triplet_loss=triplet,triplet_soft \
    dataset=cars3d,smallnorb,shapes3d,dsprites,X--xysquares \
    \
    framework=X--adanegtvae \
    \
    framework.cfg.ada_thresh_ratio=0.5 \
    framework.cfg.adat_triplet_share_scale=0.5 \
    \
    framework.cfg.triplet_margin_max=10.0 \
    framework.cfg.triplet_p=1
