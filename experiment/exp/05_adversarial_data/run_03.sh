#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-kernel-disentangle-xy"
export PARTITION="stampede"
export PARALLELISM=32
export PY_RUN_FILE='experiment/exp/05_adversarial_data/run_03_train_disentangle_kernel.py'

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
    data.name=xysquares_8x8,xysquares_4x4,xysquares_2x2,xysquares_1x1
