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

# 1 * (5 * 2*4*2) = 80
local_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_overlap_boxblur' \
    \
    +VAR.recon_loss_weight=1.0 \
    +VAR.kernel_loss_weight=3969.0 \
    +VAR.kernel_radius=31 \
    \
    run_length=medium \
    metrics=all \
    \
    dataset=X--xysquares,dsprites,shapes3d,smallnorb,cars3d \
    \
    framework=betavae,adavae_os \
    settings.framework.beta=0.0316,0.316,0.1,0.01 \
    settings.model.z_size=25,9 \
    settings.framework.recon_loss='mse_box_r${VAR.kernel_radius}_l${VAR.recon_loss_weight}_k${VAR.kernel_loss_weight}' \
    \
    sampling=default__bb
