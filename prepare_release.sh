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

function version_greater_equal() {
    printf '%s\n%s\n' "$2" "$1" | sort --check=quiet --version-sort
}

# check that we have the right version so
# that `shopt -s globstar` does not fail
if ! version_greater_equal "$BASH_VERSION" "4"; then
  echo "bash version 4 is required, got: ${BASH_VERSION}"
  exit 1
fi

# ============ #
# DELETE FILES #
# ============ #

# RESEARCH:
rm requirements-research.txt
rm requirements-research-freeze.txt
rm -rf research/

# EXPERIMENT:
rm experiment/config/config_adversarial_dataset.yaml
rm experiment/config/config_adversarial_dataset_approx.yaml
rm experiment/config/config_adversarial_kernel.yaml
rm experiment/config/run_location/griffin.yaml
rm experiment/config/run_location/heartofgold.yaml
rm experiment/config/dataset/X--*.yaml
rm experiment/config/framework/X--*.yaml

# DISENT:
# - metrics
rm disent/metrics/_flatness.py
rm disent/metrics/_flatness_components.py
# - frameworks
rm -rf disent/frameworks/ae/experimental
rm -rf disent/frameworks/vae/experimental
# - datasets
rm disent/dataset/data/_groundtruth__xcolumns.py
rm disent/dataset/data/_groundtruth__xysquares.py
rm disent/dataset/data/_groundtruth__xyblocks.py

# DATA:
# - disent.framework.helper
rm -rf data/adversarial_kernel

# TESTS:
rm tests/test_data_xy.py

# ===================== #
# DELETE LINES OF FILES #
# ===================== #

# enable recursive glob
shopt -s globstar

# scan for all files that contain 'pragma: delete-on-release'
for file in **/*.{py,yaml}; do
    if [ -n "$( grep -m 1 'pragma: delete-on-release' "$file" )" ]; then
        echo "preparing: $file"
        remove_delete_commands "$file"
    fi
done

# ===================== #
# CLEANUP THIS FILE     #
# ===================== #

rm prepare_release.sh
rm prepare_release_and_commit.sh
