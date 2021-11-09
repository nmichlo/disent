#!/bin/bash

# get the path to the script
PARENT_DIR="$(dirname "$(realpath -s "$0")")"
ROOT_DIR="$(dirname "$(dirname "$PARENT_DIR")")"

# maybe lower lr or increase batch size?
# TODO: this is out of date
PYTHONPATH="$ROOT_DIR" python3 "$PARENT_DIR/run_02_gen_adversarial_dataset_approx.py" \
    -m \
    framework.sampler_name=close_p_random_n,same_k1_close \
    framework.adversarial_mode=self,invert_margin_0.005 \
    framework.dataset_name=dsprites,shapes3d,cars3d,smallnorb
