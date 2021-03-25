#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="exp-axis-triplet"
export PARTITION="batch"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(dirname "$(realpath -s "$0")")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes batch 28800 "C-disent" # 8 hours

# 2 * (2*19=19) = 76
submit_sweep \
    +DUMMY.repeat=1,2 \
    \
    framework=X--adatvae,X--adatvae_cyclic \
    dataset=xysquares \
    run_length=short \
    \
    framework.module.triplet_margin_max=1.0 \
    framework.module.triplet_scale=0.1 \
    framework.module.triplet_p=1 \
    sampling=gt_dist_manhat \
    specializations.data_wrapper='gt_dist_${framework.data_wrap_mode}' \
    \
    framework.module.ada_triplet_ratio=1.0 \
    framework.module.triplet_mode=triplet,trip_hardAveNeg,trip_hardAveNegLerp,trip_TO_trip_hardAveNeg,trip_TO_trip_hardAveNegLerp,CONST_trip_TO_trip_hardAveNeg,CONST_trip_TO_trip_hardAveNegLerp,trip_AND_softAve,trip_AND_softAveLerp,CONST_trip_AND_softAve,CONST_trip_AND_softAveLerp,trip_scaleAve,trip_scaleAveLerp,CONST_trip_scaleAve,CONST_trip_scaleAveLerp,BROKEN_trip_scaleAve,BROKEN_trip_scaleAveLerp,CONST_BROKEN_trip_scaleAve,CONST_BROKEN_trip_scaleAveLerp

# ALL MODES:
  # triplet
  #
  # HARD AVE METHODS:
  # =================
  #     [BAD] trip_hardAveNeg
  # trip_hardAveNegLerp
  # trip_TO_trip_hardAveNeg
  # trip_TO_trip_hardAveNegLerp
  #     [BAD] CONST_trip_TO_trip_hardAveNeg
  # CONST_trip_TO_trip_hardAveNegLerp
  #
  # SOFT AVE METHODS:
  # =================
  # trip_AND_softAve
  # trip_AND_softAveLerp
  #     [BAD] CONST_trip_AND_softAve
  # CONST_trip_AND_softAveLerp
  #
  # SCALE METHODS:
  # ==============
  #     [EARLY PEAK, DECAY] trip_scaleAve
  # trip_scaleAveLerp
  #     [BAD] CONST_trip_scaleAve
  #     [EARLY PEAK, DECAY] CONST_trip_scaleAveLerp
  #
  #     [EARLY PEAK, DECAY] BROKEN_trip_scaleAve
  #     [LATE PEAK, DECAY] BROKEN_trip_scaleAveLerp
  #     [BAD] CONST_BROKEN_trip_scaleAve
  #     [EARLY PEAK, DECAY] CONST_BROKEN_trip_scaleAveLerp
