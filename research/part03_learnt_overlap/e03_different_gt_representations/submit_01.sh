#!/bin/bash


# OVERVIEW:
# - this experiment is designed to find the optimal ada-triplet function and hyper-parameters

# OUTCOMES:
# - ???


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p03e03_different-gt-representations"
export PARTITION="stampede"
export PARALLELISM=24

# the path to the generated arguments file
# - this needs to before we source the helper file
ARGS_FILE="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_01_$PROJECT.txt"

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Generate Experiment                                                       #
# ========================================================================= #

# SWEEP FOR GOOD UNSUPERVISED DO-ADA-TVAE PARAMS
#   3 * (4*2*2) = 48
ARGS_FILE="$ARGS_FILE" gen_sbatch_args_file \
    +DUMMY.repeat=1,2,3 \
    +EXTRA.tags='sweep_different-gt-repr_basic-vaes' \
    hydra.job.name="gt-repr" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.001,0.00316,0.01,0.0316 \
    framework=betavae,adavae_os \
    schedule=none \
    settings.model.z_size=9 \
    \
    dataset=xyobject,xyobject_shaded \
    sampling=default__bb


# ========================================================================= #
# Run Experiment                                                            #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 43200 "C-disent" # 12 hours

ARGS_FILE="$ARGS_FILE" submit_sbatch_args_file
