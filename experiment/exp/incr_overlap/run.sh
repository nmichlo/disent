#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-visual-overlap"
export PARTITION="batch"
export PARALLELISM=16

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

# background launch various xysquares
# 3*8=24
submit_sweep \
    framework=betavae,adavae_os,dfcvae \
    dataset=xysquares \
    dataset.data.grid_spacing=8,7,6,5,4,3,2,1

# background launch traditional datasets
# 3*4=12
submit_sweep \
    framework=betavae,adavae_os,dfcvae \
    dataset=cars3d,shapes3d,dsprites,smallnorb
