#!/bin/bash

python3 run_04_gen_adversarial_alt.py -m \
    framework.sampler_name=close_far,same_factor,random_bb \
    framework.loss_mode=self,const,invert \
    framework.dataset_name=cars3d,smallnorb
