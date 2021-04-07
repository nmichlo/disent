#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-axis-triplet-3.0"
export PARTITION="batch"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 43200 "C-disent" # 12 hours

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
    framework.module.triplet_margin_max=1.0 \
    framework.module.triplet_scale=0.1 \
    framework.module.triplet_p=1 \
    sampling=gt_dist_manhat \
    specializations.data_wrapper='gt_dist_${framework.data_wrap_mode}' \
    \
    model.z_size=25,9 \
    \
    framework.module.thresh_ratio=0.5 \
    framework.module.ada_triplet_ratio=1.0 \
    schedule=adavae_thresh,adavae_all,adavae_ratio,none \
    framework.module.ada_triplet_sample=TRUE,FALSE \
    framework.module.ada_triplet_loss=framework.module.ada_triplet_loss=triplet,triplet_soft_ave,triplet_soft_neg_ave,triplet_all_soft_ave,triplet_hard_ave,triplet_hard_neg_ave,triplet_hard_neg_ave_pull,triplet_all_hard_ave

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
