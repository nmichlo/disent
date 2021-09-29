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

# 2 * (2*4*2) = 32
submit_sweep \
    +DUMMY.repeat=1,2 \
    +EXTRA.tags='sweep_1' \
    run_length=tiny \
    -m framework=betavae,adavae_os \
    dataset=X--mask-adv-shapes3d,X--mask-ran-shapes3d,X--mask-dthr-shapes3d,shapes3d \
    model.z_size=9,25
