#!/bin/bash

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

export PROJECT="N/A"
export USERNAME="N/A"
export PARTITION="stampede"
export PARALLELISM=24

# source the helper file
source "$(dirname "$(realpath -s "$0")")/helper.sh"

# ========================================================================= #
# Experiment                                                                #
# ========================================================================= #

clog_cudaless_nodes "$PARTITION" 43200 "C-disent" # 12 hours
