#!/bin/bash

# maybe lower lr or increase batch size?
# TODO: this is out of date
python3 run_04_gen_adversarial_dataset_approx.py \
    -m \
    framework.sampler_name=close_p_random_n,same_k1_close \
    framework.adversarial_mode=self,invert_margin_0.005 \
    framework.dataset_name=dsprites,shapes3d,cars3d,smallnorb
