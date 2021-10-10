#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="test-hard-vs-soft-ada"
export PARTITION="stampede"
export PARALLELISM=16

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# 3 * (3 * 1) = 9
submit_sweep \
    +DUMMY.repeat=1,2,3 \
    +EXTRA.tags='sweep_02' \
    \
    run_length=medium \
    metrics=all \
    \
    framework.beta=1 \
    framework=adavae_os,adagvae_minimal_os,X--softadagvae_minimal_os \
    model.z_size=25 \
    \
    dataset=shapes3d \
    \
    hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97"'  # we don't want to sweep over these
