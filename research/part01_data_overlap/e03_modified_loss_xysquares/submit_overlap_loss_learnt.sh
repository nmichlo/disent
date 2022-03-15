#!/bin/bash

#
# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
# MIT License
#
# Copyright (c) 2022 Nathan Juraj Michlo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#

# OVERVIEW:
# - this experiment is designed to test how changing the reconstruction loss to match the
#   ground-truth distances allows datasets to be disentangled.


# OUTCOMES:
# - When the reconstruction loss is used as a distance function between observations, and those
#   distances match the ground truth, it enables disentanglement.
# - Loss must still be able to reconstruct the inputs correctly.
# - AEs have no incentive to learn the same distances as VAEs


# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="CVPR-09__vae_overlap_loss"
export PARTITION="stampede"
export PARALLELISM=28

# the path to the generated arguments file
# - this needs to before we source the helper file
ARGS_FILE="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")/array_overlap_learnt_${PROJECT}.txt"

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #


# TEST MSE Gaus vs MSE Learnt
# - EXTENDS: "TEST MSE vs BoxBlur MSE" from: "submit_overlap_loss.sh"
# -- in plotting, combine the results with `EXTRA.tags=="sweep_overlap_boxblur_specific"`
ARGS_FILE="$ARGS_FILE" gen_sbatch_args_file \
    +DUMMY.repeat=1,2,3,4,5 \
    +EXTRA.tags='sweep_overlap_boxblur_learnt' \
    hydra.job.name="l_ovlp_loss" \
    \
    run_length=medium \
    metrics=all \
    \
    dataset=X--xysquares \
    \
    framework=betavae,adavae_os \
    settings.framework.beta=0.0316,0.0001 \
    settings.model.z_size=25 \
    settings.framework.recon_loss='mse_gau_r31_l1.0_k3969.0','mse_xy8_r47_l1.0_k3969.0' \
    \
    sampling=default__bb


# RUN THE EXPERIMENT:
clog_cudaless_nodes "$PARTITION" 86400 "C-disent" # 24 hours
ARGS_FILE="$ARGS_FILE" submit_sbatch_args_file
