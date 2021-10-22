#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="exp-00-basic-hparam-tuning"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours

# RUN SWEEP FOR GOOD BETA VALUES
# - beta: 0.01, 0.0316 seem good, 0.1 starts getting too strong, 0.00316 is a bit weak
# - beta:
# 1 * (8 * 2 * 4) = 64
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

# RUN SWEEP FOR GOOD SCHEDULES
# 1 * (3 * 2 * 4) = 128
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_schedule' \
    \
    run_length=short \
    metrics=fast \
    \
    framework.beta=0.0316,0.1,0.316 \
    framework=betavae,adavae_os \
    schedule= \
    model.z_size=25 \
    \
    dataset=dsprites,shapes3d,cars3d,smallnorb \
    sampling=default__bb \
    \
    hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97,mscluster99"'  # we don't want to sweep over these
