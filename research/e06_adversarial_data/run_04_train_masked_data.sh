#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="exp-masked-datasets"
export PARTITION="stampede"
export PARALLELISM=20

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# 5 * (12 * 2)
submit_sweep \
    +DUMMY.repeat=1,2,3,4,5 \
    +EXTRA.tags='sweep_01' \
    \
    run_length=medium \
    \
    framework=betavae,adavae_os \
    model.z_size=9 \
    \
    dataset=X--mask-adv-dsprites,X--mask-ran-dsprites,dsprites,X--mask-adv-shapes3d,X--mask-ran-shapes3d,shapes3d,X--mask-adv-smallnorb,X--mask-ran-smallnorb,smallnorb,X--mask-adv-cars3d,X--mask-ran-cars3d,cars3d \
    specializations.dataset_sampler='random_${framework.data_sample_mode}'
