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

# ========================================================================= #
# Settings                                                                  #
# ========================================================================= #

# check variables needed in the submission script
if [ -z "$PROJECT" ];     then echo "PROJECT is not set";     exit 1; fi
if [ -z "$PARTITION" ];   then echo "PARTITION is not set";   exit 1; fi
if [ -z "$PARALLELISM" ]; then echo "PARALLELISM is not set"; exit 1; fi
if [ -z "$USERNAME" ];    then echo "USERNAME is not set";    exit 1; fi
if [ -z "$ARGS_FILE" ];   then echo "ARGS_FILE is not set";   exit 1; fi

# get the root directory
SCRIPT_DIR="$(dirname -- "${BASH_SOURCE[0]}")"
ROOT_DIR="$(realpath -s "$SCRIPT_DIR/../..")"
RUN_DIR="$ROOT_DIR/logs/sbatch_sweep"

# export variables needed in the submission script
export PROJECT
export PARTITION
export PARALLELISM
export USERNAME
export ARGS_FILE
export SCRIPT_DIR

# ========================================================================= #
# RUN FILE                                                                  #
# ========================================================================= #

echo "[LAUNCHING JOBS]:"

# change the working directory, so that logs arent written randomly everywhere
mkdir -p "$RUN_DIR"
cd "$RUN_DIR" || exit 1
echo "- working directory: $(pwd)"

# get number of lines in a file
# - the file should end with a new line
NUM_LINES=$(wc -l < "$ARGS_FILE")
echo "- submitting $NUM_LINES jobs as an array"
echo

# submit an array that reads each line from the file
# and starts a new job based on it!
sbatch \
  --partition="$PARTITION" \
  --job-name=bdisent \
  --array="1-$NUM_LINES%$PARALLELISM" \
  --time=24:00:00 \
  "$SCRIPT_DIR/sbatch_job.sh"
