#!/bin/bash

# OVERVIEW:
# - this experiment is designed to test how changing the reconstruction loss to match the
#   ground-truth distances allows datasets to be disentangled.

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p03e02_learnt-loss-with-vaes"
export PARTITION="stampede"
export PARALLELISM=24

# the path to the generated arguments file
# - this needs to before we source the helper file
ARGS_FILE="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_overlap_learnt_${PROJECT}.txt"
ARGS_FILE_RETRY="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_overlap_learnt_${PROJECT}_RETRY-2.txt"

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"


# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #


## DIFFERENT OVERLAP LOSSES
## -- changing the reconstruction loss enables disentanglement
## -- we found the optimal losses first using the metrics from
##    `part01_data_overlap/plot02_data_distances/run_data_correlation.py`
## SETUP:
## -- we mimic the run file from `part01_data_overlap/e03_modified_loss_xysquares/submit_overlap_loss.sh`
##    this experiment pretty much is a continuation
## 5 * (2*2*4 = 8) = 80
#ARGS_FILE="$ARGS_FILE" gen_sbatch_args_file \
#    +DUMMY.repeat=1,2,3,4,5 \
#    +EXTRA.tags='MSC_sweep_losses' \
#    hydra.job.name="ovlp_loss" \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    dataset=X--xysquares \
#    \
#    framework=betavae,adavae_os \
#    settings.framework.beta=0.0316,0.0001 \
#    settings.model.z_size=25 \
#    settings.framework.recon_loss='mse','mse_gau_r31_l1.0_k3969.0_norm_sum','mse_box_r31_l1.0_k3969.0_norm_sum','mse_xy8_abs63_l1.0_k1.0_norm_none' \
#    \
#    sampling=default__bb
#
#
## RUN THE EXPERIMENT:
##clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours
#ARGS_FILE="$ARGS_FILE" submit_sbatch_args_file


# -- continuation of above!
# -- frameworks crash when settings.framework.beta==0.0001, this value is too low! Try everything inbetween
#    standard MSE for some reason is fine with this value... but kernels fail...
ARGS_FILE="$ARGS_FILE_RETRY" gen_sbatch_args_file \
    +DUMMY.repeat=1,2,3,4,5 \
    +DUMMY.retry=2 \
    +EXTRA.tags='MSC_sweep_losses' \
    hydra.job.name="ovlp_loss" \
    \
    run_length=medium \
    metrics=all \
    \
    dataset=X--xysquares \
    \
    framework=betavae,adavae_os \
    settings.framework.beta=0.01,0.00316,0.001,0.000316 \
    settings.model.z_size=25 \
    settings.framework.recon_loss='mse','mse_gau_r31_l1.0_k3969.0_norm_sum','mse_box_r31_l1.0_k3969.0_norm_sum','mse_xy8_abs63_l1.0_k1.0_norm_none' \
    \
    sampling=default__bb


# RUN THE EXPERIMENT:
#clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours
ARGS_FILE="$ARGS_FILE_RETRY" submit_sbatch_args_file


# ========================================================================= #
# OLD                                                                       #
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
#    settings.framework.recon_loss='mse_gau_r31_l1.0_k3969.0_norm_sum','mse_xy8_r47_l1.0_k3969.0_norm_sum' \
#    \
#    sampling=default__bb


# RUN THE EXPERIMENT:
#clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours
#ARGS_FILE="$ARGS_FILE" submit_sbatch_args_file


# HPARAM TUNING FOR KERNEL
# -- 1 * (2*4*2*2) = 32
#ARGS_FILE="$ARGS_FILE_PARAMS" gen_sbatch_args_file \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='sweep_overlap_learnt_hparams' \
#    hydra.job.name="hparam_tune" \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    dataset=X--xysquares \
#    settings.framework.beta=0.0316,0.0001 \
#    \
#    +VAR.kernel_loss_weight=1.0,10.0,100.0,1000.0 \
#    +VAR.kernel_norm_mode=sum,abssum \
#    \
#    settings.framework.recon_loss='mse_xy8_r47_l1.0_k${VAR.kernel_loss_weight}_norm_${VAR.kernel_norm_mode}' \
#    \
#    framework=adavae_os,betavae \
#    settings.model.z_size=25 \
#    \
#    sampling=default__bb

# RUN THE EXPERIMENT:
#clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours
#ARGS_FILE="$ARGS_FILE_PARAMS" submit_sbatch_args_file


# TEST: MSE Learnt (USING ABOVE HPARAMS)
# - EXTENDS: "TEST MSE vs BoxBlur MSE" from: "submit_overlap_loss.sh"
# -- in plotting, combine the results with `EXTRA.tags=="sweep_overlap_boxblur_specific"`
# 5 * (3*2*2) = 60
#ARGS_FILE="$ARGS_FILE_TUNED" gen_sbatch_args_file \
#    +DUMMY.repeat=1,2,3,4,5 \
#    +EXTRA.tags='sweep_overlap_learnt_TUNED' \
#    hydra.job.name="l_ovlp_loss" \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    dataset=X--xysquares \
#    \
#    +VAR.kernel_loss_weight=100.0,300.0,1000.0 \
#    +VAR.kernel_norm_mode=sum \
#    \
#    framework=betavae,adavae_os \
#    settings.framework.beta=0.0316,0.0001 \
#    settings.model.z_size=25 \
#    settings.framework.recon_loss='mse_xy8_r47_l1.0_k${VAR.kernel_loss_weight}_norm_${VAR.kernel_norm_mode}' \
#    \
#    sampling=default__bb
#
## RUN THE EXPERIMENT:
##clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours
#ARGS_FILE="$ARGS_FILE_TUNED" submit_sbatch_args_file


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
