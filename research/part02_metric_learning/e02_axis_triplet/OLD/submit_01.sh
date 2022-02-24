#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export USERNAME="n_michlo"
export PROJECT="final-03__axis-triplet-3.0"
export PARTITION="batch"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(dirname "$(realpath -s "$0")")")")/scripts/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 43200 "C-disent" # 12 hours

# TODO: update this script
echo UPDATE THIS SCRIPT
exit 1

# SHORT RUNS:
# - test for best ada loss types
# 1 * (2*4*2*8=112) = 128
submit_sweep \
    +DUMMY.repeat=1 \
    \
    framework=X--adatvae \
    dataset=xysquares \
    run_length=short \
    \
    framework.cfg.triplet_margin_max=1.0 \
    framework.cfg.triplet_scale=0.1 \
    framework.cfg.triplet_p=1 \
    sampling=gt_dist_manhat \
    \
    model.z_size=25,9 \
    \
    framework.cfg.thresh_ratio=0.5 \
    framework.cfg.ada_triplet_ratio=1.0 \
    schedule=adavae_thresh,adavae_all,adavae_ratio,none \
    framework.cfg.ada_triplet_sample=TRUE,FALSE \
    framework.cfg.ada_triplet_loss=framework.cfg.ada_triplet_loss=triplet,triplet_soft_ave,triplet_soft_neg_ave,triplet_all_soft_ave,triplet_hard_ave,triplet_hard_neg_ave,triplet_hard_neg_ave_pull,triplet_all_hard_ave

# ADA TRIPLET LOSS MODES (short runs):
# - generally dont use sampling, except for: triplet_hard_neg_ave_pull
# - soft averages dont work if scheduling thresh or ratio separately, need to do both at the same time
# - hard averages perform well initially, but performance decays more toward the end of schedules
# =======================
# [X] triplet
#
# [-] triplet_soft_ave [NOTE: OK, but just worse than, triplet_all_soft_ave]
# triplet_soft_neg_ave [NOTE: better disentanglement than triplet_all_soft_ave, but worse axis align]
# triplet_all_soft_ave
#
# triplet_hard_neg_ave
# triplet_hard_neg_ave_pull     (weight = 0.1, triplet_hard_neg_ave_pull_soft)
# [X] triplet_hard_ave
# [X] triplet_hard_neg_ave_pull (weight = 1.0)
# [X] triplet_all_hard_ave
