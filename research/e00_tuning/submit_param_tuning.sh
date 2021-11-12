#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="CVPR-00__basic-hparam-tuning"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 129600 "C-disent" # 36 hours

# RUN SWEEP FOR GOOD BETA VALUES
# - beta: 0.01, 0.0316 seem good, 0.1 starts getting too strong, 0.00316 is a bit weak
# - z_size: higher means you can increase beta, eg. 25: beta=0.1 and 9: beta=0.01
# - framework: adavae needs lower beta, eg. betavae: 0.1, adavae25: 0.0316, adavae9: 0.00316
# - xy_squares really struggles to learn when non-overlapping, beta needs to be very low.
#              might be worth using a warmup schedule
#              betavae with zsize=25 and beta<=0.00316
#              betavae with zsize=09 and beta<=0.000316
#              adavae  with zsize=25 does not work
#              adavae  with zsize=09 and beta<=0.001 (must get very lucky)

# 1 * (2 * 8 * 2 * 5) = 160
submit_sweep \
    +DUMMY.repeat=1 \
    +EXTRA.tags='sweep_beta' \
    hydra.job.name="vae_hparams" \
    \
    run_length=long \
    metrics=all \
    \
    settings.framework.beta=0.000316,0.001,0.00316,0.01,0.0316,0.1,0.316,1.0 \
    framework=betavae,adavae_os \
    schedule=none \
    settings.model.z_size=9,25 \
    \
    dataset=dsprites,shapes3d,cars3d,smallnorb,X--xysquares \
    sampling=default__bb


# RUN SWEEP FOR GOOD SCHEDULES
# 1 * (3 * 2 * 4 * 5) = 120
#submit_sweep \
#    +DUMMY.repeat=1 \
#    +EXTRA.tags='sweep_schedule' \
#    \
#    run_length=long \
#    metrics=all \
#    \
#    settings.framework.beta=0.1,0.316,1.0 \
#    framework=betavae,adavae_os \
#    schedule=beta_cyclic,beta_cyclic_slow,beta_cyclic_fast,beta_decrease \
#    settings.model.z_size=25 \
#    \
#    dataset=dsprites,shapes3d,cars3d,smallnorb,X--xysquares \
#    sampling=default__bb
