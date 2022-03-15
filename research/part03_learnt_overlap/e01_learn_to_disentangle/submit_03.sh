#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p03e01_kernel-disentangle-xy"
export PARTITION="stampede"
export PARALLELISM=32

# override the default run file!
export PY_RUN_FILE='research/part03_learnt_overlap/e01_learn_to_disentangle/run_03_train_disentangle_kernel.py'

# the path to the generated arguments file
# - this needs to before we source the helper file
ARGS_FILE="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_03_$PROJECT.txt"

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

# 1 * (2*8*4) == 64
ARGS_FILE="$ARGS_FILE" gen_sbatch_args_file \
    exp.optimizer.weight_decay=0.0,1e-4 \
    exp.kernel.radius=63,55,47,39,31,23,15,7 \
    exp.data.name=xysquares_8x8,xysquares_4x4,xysquares_2x2,xysquares_1x1

# ========================================================================= #
# Run Experiment                                                            #
# ========================================================================= #

#clog_cudaless_nodes "$PARTITION" 14400 "C-disent" # 4 hours

#ARGS_FILE="$ARGS_FILE" submit_sbatch_args_file
