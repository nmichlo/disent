#!/bin/bash

# prepare the project for a new release
# removing all the research components
# - yes this is terrible, but at the rate things are changing I
#   don't want to rip things out into a separate repo... I will
#   do that eventually, but not now.

# ====== #
# HELPER #
# ====== #

function remove_delete_commands() {
  awk "!/pragma: delete-on-release/" "$1" > "$1.temp" && mv "$1.temp" "$1"
}

# ============ #
# DELETE FILES #
# ============ #

# research
rm requirements-research.txt
rm requirements-research-freeze.txt
rm -rf research/
# configs
rm experiment/config/config_adversarial_dataset.yaml
rm experiment/config/config_adversarial_dataset_approx.yaml
rm experiment/config/config_adversarial_kernel.yaml
rm experiment/config/run_location/griffin.yaml
rm experiment/config/run_location/heartofgold.yaml
rm experiment/config/dataset/X--*.yaml
rm experiment/config/dataset_sampler/episodes*.yaml
rm experiment/config/framework/X--*.yaml
# metrics
rm disent/metrics/_flatness.py
rm disent/metrics/_flatness_components.py
# frameworks
rm -rf disent/frameworks/ae/experimental
rm -rf disent/frameworks/vae/experimental
# framework helper
rm -rf data/adversarial_kernel
# datasets
rm disent/dataset/data/_groundtruth__xcolumns.py
rm disent/dataset/data/_groundtruth__xysquares.py
rm disent/dataset/data/_groundtruth__xyblocks.py
# tests
rm tests/test_data_xy.py

# ===================== #
# DELETE LINES OF FILES #
# ===================== #

# configs
remove_delete_commands experiment/config/metrics/all.yaml
remove_delete_commands experiment/config/metrics/common.yaml
remove_delete_commands experiment/config/metrics/fast.yaml
remove_delete_commands experiment/config/metrics/test.yaml
# metrics
remove_delete_commands disent/metrics/__init__.py
# framework helpers
remove_delete_commands disent/frameworks/helper/reconstructions.py
# datasets
remove_delete_commands disent/dataset/data/__init__.py
# registry
remove_delete_commands disent/registry/__init__.py
# tests
remove_delete_commands tests/test_frameworks.py
remove_delete_commands tests/test_metrics.py
