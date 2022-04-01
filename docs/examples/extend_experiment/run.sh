#!/bin/bash

# This bash script is functionally equivalent to run.py

# get the various dirs relative to this file
SCRIPT_DIR="$(realpath -s "$(dirname -- "${BASH_SOURCE[0]}")")"   # get the current script dir
DISENT_DIR="$(realpath -s "$SCRIPT_DIR/../../..")"                # get the root directory for `disent`
SEARCH_DIR="${DISENT_DIR}/docs/examples/extend_experiment/config"
RUN_SCRIPT="${DISENT_DIR}/experiment/run.py"

echo "DISENT_DIR=$DISENT_DIR"
echo "SCRIPT_DIR=$SCRIPT_DIR"
echo "SEARCH_DIR=$SEARCH_DIR"
echo "RUN_SCRIPT=$RUN_SCRIPT"

# run the experiment, passing arguments to this script to the experiment instead!
# - for example:
#   $ run.sh dataset=E--pseudorandom framework=E--si-betavae
# - is equivalent to:
#   PYTHONPATH="$DISENT_DIR" DISENT_CONFIGS_PREPEND="$SEARCH_DIR" python3 "$RUN_SCRIPT" dataset=E--pseudorandom framework=E--si-betavae
    PYTHONPATH="$DISENT_DIR" DISENT_CONFIGS_PREPEND="$SEARCH_DIR" python3 "$RUN_SCRIPT" "$@"
