#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-kernel-disentangle-xy"
export PARTITION="stampede"
export PARALLELISM=32

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# 1 * (2*8*4) == 64
submit_sweep \
    optimizer.weight_decay=1e-4,0.0 \
    kernel.radius=63,55,47,39,31,23,15,7 \
    dataset.spacing=8,4,2,1 \
