#!/bin/bash

# get the path to the script
PARENT_DIR="$(dirname "$(realpath -s "$0")")"
ROOT_DIR="$(dirname "$(dirname "$(dirname "$PARENT_DIR")")")"

# TODO: fix this!
# TODO: this is out of date
PYTHONPATH="$ROOT_DIR" python3 "$PARENT_DIR/run_02_gen_adversarial_dataset.py" \
    -m \
    framework.sampler_name=same_k,close_far,same_factor,random_bb \
    framework.loss_mode=self,const,invert \
    framework.dataset_name=cars3d,smallnorb
