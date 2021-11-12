#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="CVPR-01__incr_overlap"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours


# background launch various xysquares
# -- original experiment also had dfcvae
# -- beta is too high for adavae
# 3 * (2*2*8 = 32) = 96
submit_sweep \
    +DUMMY.repeat=1,2,3 \
    +EXTRA.tags='sweep_xy_squares_overlap' \
    hydra.job.name="incr_ovlp" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.001,0.00316 \
    framework=betavae,adavae_os \
    settings.model.z_size=9 \
    \
    sampling=default__bb \
    dataset=X--xysquares_rgb \
    dataset.data.grid_spacing=8,7,6,5,4,3,2,1


# background launch various xysquares
# - this time we try delay beta, so that it can learn properly...
# 3 * (2*2*8 = 32) = 96
submit_sweep \
    +DUMMY.repeat=1,2,3 \
    +EXTRA.tags='sweep_xy_squares_overlap_delay' \
    hydra.job.name="schd_incr_ovlp" \
    \
    schedule=beta_delay_long \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.001 \
    framework=betavae,adavae_os \
    settings.model.z_size=9,25 \
    \
    sampling=default__bb \
    dataset=X--xysquares_rgb \
    dataset.data.grid_spacing=8,7,6,5,4,3,2,1


# background launch traditional datasets
# -- original experiment also had dfcvae
# 5 * (2*2*4 = 16) = 80
#submit_sweep \
#    +DUMMY.repeat=1,2,3,4,5 \
#    +EXTRA.tags='sweep_other' \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    settings.framework.beta=0.01,0.0316 \
#    framework=betavae,adavae_os \
#    settings.model.z_size=9 \
#    \
#    sampling=default__bb \
#    dataset=cars3d,shapes3d,dsprites,smallnorb
