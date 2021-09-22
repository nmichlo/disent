#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-masked-datasets"
export PARTITION="stampede"
export PARALLELISM=30

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# 2 * (2*8) = 32
submit_sweep \
    +DUMMY.repeat=1,2 \
    +EXTRA.tags='run-1' \
    run_length=short \
    dataset=cars3d,dsprites,X--mask-adv-cars3d,X--mask-adv-smallnorb,X--mask-dthr-cars3d,X--mask-dthr-smallnorb,X--mask-ran-cars3d,X--mask-ran-smallnorb \
    model.z_size=9 \
    framework=betavae,adavae_os \
    specializations.dataset_sampler='random_${framework.data_sample_mode}'
