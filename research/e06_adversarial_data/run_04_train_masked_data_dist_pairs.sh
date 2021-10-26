#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="exp-masked-datasets-dist-pairs"
export PARTITION="stampede"
export PARALLELISM=36

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# (3*2*3*12 = 72) = 216
# TODO: z_size needs tuning
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_dist_pairs_usage_ratio' \
    \
    run_callbacks=vis \
    run_length=short \
    metrics=all \
    \
    framework.beta=0.0316,0.01,0.1 \
    framework=betavae,adavae_os \
    model.z_size=16 \
    framework.optional.usage_ratio=0.5,0.2,0.05 \
    \
    dataset=X--mask-adv-r-dsprites,X--mask-ran-dsprites,dsprites,X--mask-adv-r-shapes3d,X--mask-ran-shapes3d,shapes3d,X--mask-adv-r-smallnorb,X--mask-ran-smallnorb,smallnorb,X--mask-adv-r-cars3d,X--mask-ran-cars3d,cars3d \
    sampling=random \
    \
    hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97,mscluster99"'  # we don't want to sweep over these
