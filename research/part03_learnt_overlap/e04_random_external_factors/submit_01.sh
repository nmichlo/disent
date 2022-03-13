#!/bin/bash


# OVERVIEW:
# - this experiment is designed to find the optimal ada-triplet function and hyper-parameters

# OUTCOMES:
# - ???


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="MSC-p03e04_random-external-factors"
export PARTITION="stampede"
export PARALLELISM=28

# the path to the generated arguments file
# - this needs to before we source the helper file
ARGS_FILE="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_01_${PROJECT}_FIX.txt"

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Generate Experiment                                                       #
# ========================================================================= #

# SWEEP FOR GOOD PARAMS
#   1 * (2*3*2*9) = 108
ARGS_FILE="$ARGS_FILE" gen_sbatch_args_file \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_imagenet_dsprites_FIX' \
    hydra.job.name="imnet-dsprites" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.01,0.00316,0.0316 \
    framework=betavae,adavae_os \
    schedule=none \
    settings.model.z_size=9 \
    \
    dataset=dsprites,X--dsprites-imagenet-bg-25,X--dsprites-imagenet-bg-50,X--dsprites-imagenet-bg-75,X--dsprites-imagenet-bg-100,X--dsprites-imagenet-fg-25,X--dsprites-imagenet-fg-50,X--dsprites-imagenet-fg-75,X--dsprites-imagenet-fg-100 \
    sampling=default__bb


# ========================================================================= #
# Run Experiment                                                            #
# ========================================================================= #

#clog_cudaless_nodes "$PARTITION" 43200 "C-disent" # 12 hours

ARGS_FILE="$ARGS_FILE" submit_sbatch_args_file
