#!/bin/bash


# OVERVIEW:
# - this experiment is designed to find the correlation between our proposed
#   "factored components" metrics and beta values of the frameworks.


# OUTCOMES:


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p02e00_beta-data-latent-corr"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours


# -- metrics in the original experiment were updated after the runs
#    we could have reused those results, but we needed the updated metric values...
# 1 * (2 * 9 * 2 * 5) = 180
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_beta_corr' \
    hydra.job.name="vae_beta" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.optimizer.lr=1e-3,1e-4
    settings.framework.beta=0.0001,0.000316,0.001,0.00316,0.01,0.0316,0.1,0.316,1.0 \
    framework=betavae,adavae_os \
    schedule=none \
    settings.model.z_size=25 \
    \
    dataset=dsprites,shapes3d,cars3d,smallnorb,X--xysquares \
    sampling=default__bb
