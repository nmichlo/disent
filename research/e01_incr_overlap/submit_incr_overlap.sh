#!/bin/bash


# OVERVIEW:
# - this experiment is designed to check how increasing overlap (reducing
#   the spacing between square positions on XYSquares) affects learning.


# OUTCOMES:
# - increasing overlap improves disentanglement & ability for the
#   neural network to learn values.
# - decreasing overlap worsens disentanglement, but it also becomes
#    very hard for the neural net to learn specific values needed. The
#    average image does not correspond well to individual samples.
#    Disentanglement performance is also a result of this fact, as
#    the network can't always learn the dataset effectively.


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #


export USERNAME="n_michlo"
export PROJECT="CVPR-01__incr_overlap"
export PARTITION="stampede"
export PARALLELISM=28

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"


# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #


clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours


# background launch various xysquares
# -- original experiment also had dfcvae
# -- beta is too high for adavae
# 5 * (2*2*8 = 32) = 160
submit_sweep \
    +DUMMY.repeat=1,2,3,4,5 \
    +EXTRA.tags='sweep_xy_squares_overlap' \
    hydra.job.name="incr_ovlp" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.001,0.00316 \
    framework=betavae,adavae_os \
    settings.model.z_size=9 \
    \
    sampling=default__bb \
    dataset=X--xysquares_rgb \
    dataset.data.grid_spacing=8,7,6,5,4,3,2,1


# background launch various xysquares
# -- original experiment also had dfcvae
# -- beta is too high for adavae
# 5 * (2*8 = 16) = 80
submit_sweep \
    +DUMMY.repeat=1,2,3,4,5 \
    +EXTRA.tags='sweep_xy_squares_overlap_small_beta' \
    hydra.job.name="sb_incr_ovlp" \
    \
    run_length=medium \
    metrics=all \
    \
    settings.framework.beta=0.0001,0.00001 \
    framework=adavae_os \
    settings.model.z_size=9 \
    \
    sampling=default__bb \
    dataset=X--xysquares_rgb \
    dataset.data.grid_spacing=8,7,6,5,4,3,2,1


# background launch various xysquares
# - this time we try delay beta, so that it can learn properly...
# - NOTE: this doesn't actually work, the VAE loss often becomes
#         NAN because the values are too small.
# 3 * (2*2*8 = 32) = 96
# submit_sweep \
#     +DUMMY.repeat=1,2,3 \
#     +EXTRA.tags='sweep_xy_squares_overlap_delay' \
#     hydra.job.name="schd_incr_ovlp" \
#     \
#     schedule=beta_delay_long \
#     \
#     run_length=medium \
#     metrics=all \
#     \
#     settings.framework.beta=0.001 \
#     framework=betavae,adavae_os \
#     settings.model.z_size=9,25 \
#     \
#     sampling=default__bb \
#     dataset=X--xysquares_rgb \
#     dataset.data.grid_spacing=8,7,6,5,4,3,2,1


# background launch traditional datasets
# -- original experiment also had dfcvae
# 5 * (2*2*4 = 16) = 80
#submit_sweep \
#    +DUMMY.repeat=1,2,3,4,5 \
#    +EXTRA.tags='sweep_other' \
#    \
#    run_length=medium \
#    metrics=all \
#    \
#    settings.framework.beta=0.01,0.0316 \
#    framework=betavae,adavae_os \
#    settings.model.z_size=9 \
#    \
#    sampling=default__bb \
#    dataset=cars3d,shapes3d,dsprites,smallnorb


# ========================================================================= #
# DONE                                                                      #
# ========================================================================= #
