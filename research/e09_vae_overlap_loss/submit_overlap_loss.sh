#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="CVPR-09__vae_overlap_loss"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# TEST MSE vs BoxBlur MSE (with different beta values over different datasets)
# - mse boxblur weight is too strong, need to lower significantly
# 1 * (5 * 2*4*2) = 80
#submit_sweep \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='sweep_overlap_boxblur' \
#    hydra.job.name="ovlp_loss" \
#    \
#    +VAR.recon_loss_weight=1.0 \
#    +VAR.kernel_loss_weight=3969.0 \
#    +VAR.kernel_radius=31 \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    dataset=X--xysquares,dsprites,shapes3d,smallnorb,cars3d \
#    \
#    framework=betavae,adavae_os \
#    settings.framework.beta=0.0316,0.316,0.1,0.01 \
#    settings.model.z_size=25,9 \
#    settings.framework.recon_loss='mse_box_r${VAR.kernel_radius}_l${VAR.recon_loss_weight}_k${VAR.kernel_loss_weight}' \
#    \
#    sampling=default__bb


# TEST MSE vs BoxBlur MSE
# - changing the reconstruction loss enables disentanglement
# 5 * (2*2*2 = 8) = 40
submit_sweep \
    +DUMMY.repeat=1,2,3,4,5 \
    +EXTRA.tags='sweep_overlap_boxblur_specific' \
    hydra.job.name="s_ovlp_loss" \
    \
    +VAR.recon_loss_weight=1.0 \
    +VAR.kernel_loss_weight=3969.0 \
    +VAR.kernel_radius=31 \
    \
    run_length=medium \
    metrics=all \
    \
    dataset=X--xysquares \
    \
    framework=betavae,adavae_os \
    settings.framework.beta=0.0316,0.0001 \
    settings.model.z_size=25 \
    settings.framework.recon_loss=mse,'mse_box_r${VAR.kernel_radius}_l${VAR.recon_loss_weight}_k${VAR.kernel_loss_weight}' \
    \
    sampling=default__bb


# TEST DISTANCES IN AEs VS VAEs
# -- supplementary material
# 3 * (1 * 2 = 2) = 6
submit_sweep \
    +DUMMY.repeat=1,2,3 \
    +EXTRA.tags='sweep_overlap_boxblur_autoencoders' \
    hydra.job.name="e_ovlp_loss" \
    \
    +VAR.recon_loss_weight=1.0 \
    +VAR.kernel_loss_weight=3969.0 \
    +VAR.kernel_radius=31 \
    \
    run_length=medium \
    metrics=all \
    \
    dataset=X--xysquares \
    \
    framework=ae \
    settings.framework.beta=0.0001 \
    settings.model.z_size=25 \
    settings.framework.recon_loss=mse,'mse_box_r${VAR.kernel_radius}_l${VAR.recon_loss_weight}_k${VAR.kernel_loss_weight}' \
    \
    sampling=default__bb


# HPARAM SWEEP -- TODO: update
# -- old, unused
# 1 * (2 * 8 * 2 * 2) = 160
#submit_sweep \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='sweep_beta' \
#    hydra.job.name="vae_hparams" \
#    \
#    run_length=long \
#    metrics=all \
#    \
#    settings.framework.beta=0.000316,0.001,0.00316,0.01,0.0316,0.1,0.316,1.0 \
#    framework=betavae,adavae_os \
#    schedule=none \
#    settings.model.z_size=9,25 \
#    \
#    dataset=X--xysquares \
#    sampling=default__bb
