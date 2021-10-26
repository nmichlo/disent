#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-06__adversarial-modified-data"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# TODO: update this script
echo UPDATE THIS SCRIPT
exit 1

# 1 * (4 * 2 * 2) = 16
local_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_griffin' \
    run_location='griffin' \
    \
    run_length=short \
    metrics=fast \
    \
    framework.beta=0.001,0.00316,0.01,0.000316 \
    framework=betavae,adavae_os \
    model.z_size=25 \
    \
    dataset=X--adv-dsprites--WARNING,X--adv-shapes3d--WARNING \
    sampling=default__bb # \
    # \
    # hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97"'  # we don't want to sweep over these

# 2 * (8 * 2 * 4) = 128
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_beta' \
    \
    run_length=short \
    metrics=fast \
    \
    framework.beta=0.000316,0.001,0.00316,0.01,0.0316,0.1,0.316,1.0 \
    framework=betavae,adavae_os \
    model.z_size=25 \
    \
    dataset=dsprites,shapes3d,cars3d,smallnorb \
    sampling=default__bb \
    \
    hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97,mscluster99"'  # we don't want to sweep over these
