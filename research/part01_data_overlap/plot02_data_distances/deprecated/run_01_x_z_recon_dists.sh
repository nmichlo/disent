#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-01__gt-vs-learnt-dists"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours


# 1 * (3 * 6 * 4 * 2) = 144
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep' \
    \
    model=linear,vae_fc,vae_conv64 \
    \
    run_length=medium \
    metrics=all \
    \
    dataset=xyobject,xyobject_shaded,shapes3d,dsprites,cars3d,smallnorb \
    sampling=default__bb \
    framework=ae,X--adaae_os,betavae,adavae_os \
    \
    settings.framework.beta=0.0316 \
    settings.optimizer.lr=3e-4 \
    settings.model.z_size=9,25
