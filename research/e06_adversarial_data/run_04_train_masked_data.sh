#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="exp-masked-datasets"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# 3 * (12 * 2 * 2) = 144
#submit_sweep \
#    +DUMMY.repeat=1,2,3 \
#    +EXTRA.tags='sweep_01' \
#    \
#    run_length=medium \
#    \
#    framework.beta=0.001 \
#    framework=betavae,adavae_os \
#    model.z_size=9 \
#    \
#    dataset=X--mask-adv-dsprites,X--mask-ran-dsprites,dsprites,X--mask-adv-shapes3d,X--mask-ran-shapes3d,shapes3d,X--mask-adv-smallnorb,X--mask-ran-smallnorb,smallnorb,X--mask-adv-cars3d,X--mask-ran-cars3d,cars3d \
#    sampling=random \
#    \
#    hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97"'  # we don't want to sweep over these

# 3 * (12 * 4 * 2) = 288
submit_sweep \
    +DUMMY.repeat=1,2,3 \
    +EXTRA.tags='sweep_usage_ratio' \
    \
    run_length=short \
    metrics=all_fast_final \
    \
    framework.beta=0.001 \
    framework=betavae,adavae_os \
    model.z_size=25 \
    framework.optional.usage_ratio=0.5,0.25,0.1,0.05 \
    \
    dataset=X--mask-adv-dsprites,X--mask-ran-dsprites,dsprites,X--mask-adv-shapes3d,X--mask-ran-shapes3d,shapes3d,X--mask-adv-smallnorb,X--mask-ran-smallnorb,smallnorb,X--mask-adv-cars3d,X--mask-ran-cars3d,cars3d \
    sampling=random \
    \
    hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97"'  # we don't want to sweep over these
