#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="exp-dsprites-imagenet"
export PARTITION="stampede"
export PARALLELISM=36

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# (3*2*2*11) = 132
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_dsprites_imagenet' \
    \
    run_callbacks=vis \
    run_length=medium \
    metrics=fast \
    \
    model.z_size=9,16 \
    framework.beta=0.0316,0.01,0.1 \
    framework=adavae_os,betavae \
    \
    dataset=dsprites,X--dsprites-imagenet-bg-20,X--dsprites-imagenet-bg-40,X--dsprites-imagenet-bg-60,X--dsprites-imagenet-bg-80,X--dsprites-imagenet-bg-100,X--dsprites-imagenet-fg-20,X--dsprites-imagenet-fg-40,X--dsprites-imagenet-fg-60,X--dsprites-imagenet-fg-80,X--dsprites-imagenet-fg-100 \
    sampling=default__bb \
    \
    hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97,mscluster99"'  # we don't want to sweep over these
