#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-00-basic-hparam-tuning"
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
# 1 * (2 * 8 * 2 * 4) = 128
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_beta' \
    \
    run_length=short \
    metrics=fast \
    \
    framework=betavae,adavae_os \
    settings.framework.beta=0.000316,0.001,0.00316,0.01,0.0316,0.1,0.316,1.0 \
    settings.model.z_size=9,25 \
    \
    dataset=dsprites,shapes3d,cars3d,smallnorb \
    sampling=default__bb

# RUN SWEEP FOR GOOD SCHEDULES
# 1 * (4 * 2 * 4 * 5) = 160
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_schedule' \
    \
    run_length=long \
    metrics=fast \
    \
    settings.framework.beta=0.0316,0.1,0.316,1.0 \
    framework=betavae,adavae_os \
    schedule=beta_cyclic,beta_cyclic_slow,beta_cyclic_fast,beta_decrease \
    settings.model.z_size=9,25 \
    \
    dataset=dsprites,shapes3d,cars3d,smallnorb \
    sampling=default__bb
