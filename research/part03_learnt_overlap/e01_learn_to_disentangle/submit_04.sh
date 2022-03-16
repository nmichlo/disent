#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p03e02_disentangle-data"
export PARTITION="stampede"
export PARALLELISM=32

# override the default run file!
export PY_RUN_FILE='research/part03_learnt_overlap/e01_learn_to_disentangle/run_04_train_disentangle_model.py'

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

# 1 * (2*8*4) == 64
local_sweep \
    +EXTRA.tags='test' \
    \
    dis_system.loss_disentangle_weight=1.0 \
    dis_system.loss_out_of_bounds_weight=1.0 \
    dis_system.loss_stats_mean_weight=NULL \
    dis_system.loss_stats_var_weight=1.0 \
    dis_system.loss_similarity_weight=1.0 \
    \
    dis_system.disentangle_combined_loss=FALSE \
    dis_system.disentangle_mode=invert,improve \
    dis_system.dataset_name=cars3d,smallnorb,dsprites,shapes3d,xysquares_8x8
