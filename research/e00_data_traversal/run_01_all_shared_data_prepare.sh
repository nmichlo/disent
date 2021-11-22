#!/bin/bash

# This script is intended to prepare all shared data on the wits cluster
# you can probably modify it for your own purposes
# - data is loaded and processed into ~/downloads/datasets which is a
#   shared drive, instead of /tmp/<user>, which is a local drive.

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="prepare-data"
export PARTITION="stampede"
export PARALLELISM=32

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

DATASETS=(
  cars3d
  dsprites
  # monte_rollouts
  # mpi3d_real
  # mpi3d_realistic
  # mpi3d_toy
  shapes3d
  smallnorb
  #X--adv-cars3d--WARNING
  #X--adv-dsprites--WARNING
  #X--adv-shapes3d--WARNING
  #X--adv-smallnorb--WARNING
  #X--dsprites-imagenet
  #X--dsprites-imagenet-bg-20
  #X--dsprites-imagenet-bg-40
  #X--dsprites-imagenet-bg-60
  #X--dsprites-imagenet-bg-80
  #X--dsprites-imagenet-bg-100
  #X--dsprites-imagenet-fg-20
  #X--dsprites-imagenet-fg-40
  #X--dsprites-imagenet-fg-60
  #X--dsprites-imagenet-fg-80
  #X--dsprites-imagenet-fg-100
  #X--mask-adv-f-cars3d
  #X--mask-adv-f-dsprites
  #X--mask-adv-f-shapes3d
  #X--mask-adv-f-smallnorb
  #X--mask-adv-r-cars3d
  #X--mask-adv-r-dsprites
  #X--mask-adv-r-shapes3d
  #X--mask-adv-r-smallnorb
  #X--mask-dthr-cars3d
  #X--mask-dthr-dsprites
  #X--mask-dthr-shapes3d
  #X--mask-dthr-smallnorb
  #X--mask-ran-cars3d
  #X--mask-ran-dsprites
  #X--mask-ran-shapes3d
  #X--mask-ran-smallnorb
  "X--xyblocks"
  #X--xyblocks_grey
  "X--xysquares"
  #X--xysquares_grey
  #X--xysquares_rgb
  xyobject
  #xyobject_grey
  #xyobject_shaded
  #xyobject_shaded_grey
)

local_sweep \
    run_action=prepare_data \
    run_location=stampede_shr \
    run_launcher=local \
    dataset="$(IFS=, ; echo "${DATASETS[*]}")"
