#!/bin/bash

# OVERVIEW:
# - this experiment is designed to test how changing the reconstruction loss to match the
#   ground-truth distances allows datasets to be disentangled.


# OUTCOMES:
# - When the reconstruction loss is used as a distance function between observations, and those
#   distances match the ground truth, it enables disentanglement.
# - Loss must still be able to reconstruct the inputs correctly.
# - AEs have no incentive to learn the same distances as VAEs


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="CVPR-09__vae_overlap_loss"
export PARTITION="stampede"
export PARALLELISM=24

# the path to the generated arguments file
# - this needs to before we source the helper file
ARGS_FILE="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_overlap_learnt_${PROJECT}.txt"
ARGS_FILE_PARAMS="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_overlap_learnt_${PROJECT}_PARAMS.txt"

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #


# TEST MSE Gaus vs MSE Learnt
# - EXTENDS: "TEST MSE vs BoxBlur MSE" from: "submit_overlap_loss.sh"
# -- in plotting, combine the results with `EXTRA.tags=="sweep_overlap_boxblur_specific"`
#ARGS_FILE="$ARGS_FILE" gen_sbatch_args_file \
#    +DUMMY.repeat=1,2,3,4,5 \
#    +EXTRA.tags='sweep_overlap_boxblur_learnt' \
#    hydra.job.name="l_ovlp_loss" \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    dataset=X--xysquares \
#    \
#    framework=betavae,adavae_os \
#    settings.framework.beta=0.0316,0.0001 \
#    settings.model.z_size=25 \
#    settings.framework.recon_loss='mse_gau_r31_l1.0_k3969.0','mse_xy8_r47_l1.0_k3969.0' \
#    \
#    sampling=default__bb


# RUN THE EXPERIMENT:
#clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours
#ARGS_FILE="$ARGS_FILE" submit_sbatch_args_file


# HPARAM TUNING FOR KERNEL
# -- 1 * (2*4*2*2) = 32
ARGS_FILE="$ARGS_FILE_PARAMS" gen_sbatch_args_file \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_overlap_learnt_hparams' \
    hydra.job.name="hparam_tune" \
    \
    run_length=medium \
    metrics=all \
    \
    dataset=X--xysquares \
    settings.framework.beta=0.0316,0.0001 \
    \
    +VAR.kernel_loss_weight=1.0,10.0,100.0,1000.0 \
    +VAR.kernel_norm_mode=sum,abssum \
    \
    settings.framework.recon_loss='mse_xy8_r47_l1.0_k${VAR.kernel_loss_weight}_norm_${VAR.kernel_norm_mode}' \
    \
    framework=adavae_os,betavae \
    settings.model.z_size=25 \
    \
    sampling=default__bb


# RUN THE EXPERIMENT:
clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours
ARGS_FILE="$ARGS_FILE_PARAMS" submit_sbatch_args_file
