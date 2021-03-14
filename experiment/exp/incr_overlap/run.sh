#!/bin/bash

# get the root directory
SCRIPT_DIR=$(dirname "$(realpath -s "$0")")
ROOT_DIR=$(realpath -s "$SCRIPT_DIR/../../..")

# cd into the root, exit on failure
cd "$ROOT_DIR" || exit 1
echo "working directory is: $(pwd)"

# background launch various xysquares
PYTHONPATH="$ROOT_DIR" python3 experiment/run.py -m hydra.launcher.array_parallelism=16 framework=betavae,adavae_os,dfcvae dataset=xysquares dataset.data.grid_spacing=8,7,6,5,4,3,2,1 &
# background launch traditional datasets
PYTHONPATH="$ROOT_DIR" python3 experiment/run.py -m hydra.launcher.array_parallelism=16 framework=betavae,adavae_os,dfcvae dataset=cars3d,shapes3d,dsprites,smallnorb &
