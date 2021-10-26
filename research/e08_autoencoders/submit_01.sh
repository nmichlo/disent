#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-08__autoencoder-versions"
export PARTITION="stampede"
export PARALLELISM=32

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# 1 * (2*2*3*3*8) == 288
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='various-auto-encoders' \
    \
    run_length=short,long \
    schedule=adavae_up_ratio_full,adavae_up_all_full,none \
    \
    dataset=xysquares,cars3d,shapes3d \
    framework=ae,tae,X--adaae,X--adanegtae,vae,tvae,adavae,X--adanegtvae \
    model=conv64alt \
    model.z_size=25 \
    \
    sampling=gt_dist_manhat,gt_dist_manhat_scaled
