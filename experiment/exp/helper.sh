#!/bin/bash

# ========================================================================= #
# Description                                                               #
# ========================================================================= #

# before sourcing this script, it requires the following variables to be exported:
# - PROJECT: str
# - PARTITION: str
# - PARALLELISM: int

# source this script from the script you use to run the experiment
# 1. gets and exports the path to the root
# 2. changes the working directory to the root
# 3. exports a helper function that runs the script in the background, with
#    the correct python path and settings

if [ -z ${PROJECT+x} ]; then echo "PROJECT is not set"; exit 1; fi
if [ -z ${PARTITION+x} ]; then echo "PARTITION is not set"; exit 1; fi
if [ -z ${PARALLELISM+x} ]; then echo "PARALLELISM is not set"; exit 1; fi

# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #

# get the root directory
SCRIPT_DIR=$(dirname "$(realpath -s "$0")")
ROOT_DIR="$(realpath -s "$SCRIPT_DIR/../../..")"

# cd into the root, exit on failure
cd "$ROOT_DIR" || exit 1
echo "working directory is: $(pwd)"

function submit_sweep() {
    echo "SUBMITTING:" "$@"
    PYTHONPATH="$ROOT_DIR" python3 experiment/run.py -m \
        job.project="$PROJECT" \
        job.partition="$PARTITION" \
        hydra.launcher.array_parallelism="$PARALLELISM" \
        "$@" \
        & # run in background
}

function local_run() {
    echo "RUNNING:" "$@"
    PYTHONPATH="$ROOT_DIR" python3 experiment/run.py \
        job.project="$PROJECT" \
        "$@"
}

# export
export ROOT_DIR
export submit_sweep
export local_run

# ========================================================================= #
# End                                                                       #
# ========================================================================= #
