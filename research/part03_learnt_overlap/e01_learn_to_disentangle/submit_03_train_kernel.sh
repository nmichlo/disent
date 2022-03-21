#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p03e01_kernel-disentangle-xy"
export PARTITION="stampede"
export PARALLELISM=24

# override the default run file!
export PY_RUN_FILE='research/part03_learnt_overlap/e01_learn_to_disentangle/run_03_train_disentangle_kernel.py'

# the path to the generated arguments file
# - this needs to before we source the helper file
ARGS_FILE="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_03_$PROJECT.txt"
ARGS_FILE_TUNED="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_03_${PROJECT}_TUNED.txt"

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

# 1 * (4*2*4*4) == 128
ARGS_FILE="$ARGS_FILE" gen_sbatch_args_file \
    +EXTRA.tags='sweep' \
    hydra.job.name="kernel" \
    \
    run_length=short \
    settings.job.name_prefix="MSC" \
    \
    settings.dataset.batch_size=512 \
    exp.kernel.represent_mode=abs,square,exp,none \
    exp.optimizer.lr=1e-3,5e-4 \
    exp.optimizer.weight_decay=0.0 \
    \
    exp.data.name=xysquares_8x8,xysquares_4x4,xysquares_2x2,xysquares_1x1 \
    exp.kernel.radius=63,47,31,15 \

# 63,55,47,39,31,23,15,7
#clog_cudaless_nodes "$PARTITION" 14400 "C-disent" # 4 hours
ARGS_FILE="$ARGS_FILE" submit_sbatch_args_file


# 1 * (2*2*4) = 16
ARGS_FILE="$ARGS_FILE_TUNED" gen_sbatch_args_file \
    +EXTRA.tags='sweep_MED' \
    hydra.job.name="kernel" \
    \
    run_length=medium \
    settings.job.name_prefix="MSC_TUNED" \
    \
    settings.dataset.batch_size=512 \
    exp.kernel.represent_mode=abs,none \
    exp.optimizer.lr=1e-3,5e-4 \
    exp.optimizer.weight_decay=0.0 \
    \
    exp.data.name=xysquares_8x8 \
    exp.kernel.radius=63,47,31,15 \


#clog_cudaless_nodes "$PARTITION" 14400 "C-disent" # 4 hours
ARGS_FILE="$ARGS_FILE_TUNED" submit_sbatch_args_file


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
