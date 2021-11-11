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

# RUN SWEEP FOR GOOD BETA VALUES
# - beta: 0.01, 0.0316 seem good, 0.1 starts getting too strong, 0.00316 is a bit weak
# - beta:
# 1 * (2 * 5 * 2 * 6) = 120
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_overlap_loss' \
    \
    run_length=long \
    metrics=all \
    \
    framework=betavae,adavae_os \
    settings.framework.beta=0.001,0.00316,0.01,0.0316,0.1 \
    settings.model.z_size=9,25 \
    settings.framework.recon_loss=mse_box_r47_w0.5,mse_box_r45_w0.5,mse_box_r33_w0.5,mse_box_r29_w0.5,mse_box_r25_w0.5,mse \
    \
    dataset=X--xysquares \
    sampling=default__bb
