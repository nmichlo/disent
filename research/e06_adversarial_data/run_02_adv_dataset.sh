#!/bin/bash

# TODO: fix this!
# TODO: this is out of date
python3 run_02_gen_adversarial_dataset.py \
    -m \
    framework.sampler_name=same_k,close_far,same_factor,random_bb \
    framework.loss_mode=self,const,invert \
    framework.dataset_name=cars3d,smallnorb
