#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="exp-gt-vs-learnt-dists"
export PARTITION="stampede"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours


# 3 * (4 * 6) = 72
submit_sweep \
    +DUMMY.repeat=1,2,3 \
    +EXTRA.tags='sweep_01' \
    \
    run_length=short \
    metrics=fast \
    \
    dataset=xyobject,xyobject_shaded,shapes3d,dsprites,cars3d,smallnorb \
    specializations.dataset_sampler='${dataset.data_type}__${framework.data_sample_mode}' \
    \
    framework.beta=0.001 \
    framework=ae,X--adaae_os,betavae,adavae_os \
    dataset_sampler_cfg=gt__bb \
    model.z_size=25 \
    \
    hydra.launcher.exclude='"mscluster93,mscluster94,mscluster97"'  # we don't want to sweep over these
