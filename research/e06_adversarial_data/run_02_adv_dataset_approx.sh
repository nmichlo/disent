#!/bin/bash

# get the path to the script
PARENT_DIR="$(dirname "$(realpath -s "$0")")"
ROOT_DIR="$(dirname "$(dirname "$PARENT_DIR")")"

# maybe lower lr or increase batch size?
#PYTHONPATH="$ROOT_DIR" python3 "$PARENT_DIR/run_02_gen_adversarial_dataset_approx.py" \
#    -m \
#    adv_system.sampler_name=close_p_random_n,same_k1_close \
#    adv_system.adversarial_mode=self,invert_margin_0.005 \
#    adv_system.dataset_name=dsprites,shapes3d,cars3d,smallnorb

#PYTHONPATH="$ROOT_DIR" python3 "$PARENT_DIR/run_02_gen_adversarial_dataset_approx.py" \
#    -m \
#    settings.dataset.batch_size=32,256 \
#    adv_system.loss_out_of_bounds_weight=0.0,1.0 \
#    \
#    adv_system.sampler_name=close_p_random_n \
#    adv_system.adversarial_mode=invert_margin_0.05,invert_margin_0.005,invert_margin_0.0005 \
#    adv_system.dataset_name=smallnorb

PYTHONPATH="$ROOT_DIR" python3 "$PARENT_DIR/run_02_gen_adversarial_dataset_approx.py" \
    -m "$@" \
    settings.dataset.batch_size=128 \
    adv_system.loss_out_of_bounds_weight=1.0 \
    \
    adv_system.sampler_name=same_k1_close,close_p_random_n,random_swap_manhattan \
    adv_system.samples_sort_mode=swap,sort_reverse,none,sort_inorder \
    \
    adv_system.adversarial_mode=triplet_margin_0.1 \
    adv_system.dataset_name=smallnorb \
