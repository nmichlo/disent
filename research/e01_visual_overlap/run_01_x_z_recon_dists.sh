#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="exp-gt-vs-learnt-dists"
export PARTITION="stampede"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours


# 1 * (3 * 4 * 6) = 72
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_01' \
    \
    model=linear,vae_fc,vae_conv64
    \
    run_length=short \
    metrics=fast \
    \
    dataset=xyobject,xyobject_shaded,shapes3d,dsprites,cars3d,smallnorb \
    sampling=default__bb \
    \
    framework.beta=0.001 \
    framework=ae,X--adaae_os,betavae,adavae_os \
    model.z_size=25 \
    \
    hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97"'  # we don't want to sweep over these
