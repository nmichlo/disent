#!/bin/bash


# ====== #
# HELPER #
# ====== #

function version_greater_equal() {
    printf '%s\n%s\n' "$2" "$1" | sort --check=quiet --version-sort
}

# check that we have the right version so
# that `shopt -s globstar` does not fail
if ! version_greater_equal "$BASH_VERSION" "4"; then
  echo "bash version 4 is required, got: ${BASH_VERSION}"
  exit 1
fi

# ====== #
# RUN    #
# ====== #

echo "(1/3) [GIT] Creating Prepare Branch" && \
    git checkout -b xdev-prepare && \
    ( git branch --unset-upstream 2>/dev/null || true ) && \
    \
    echo "(2/3) [PREPARE]" && \
    bash ./prepare_release.sh && \
    \
    echo "(3/3) [GIT] Committing Files" && \
    git add .  && \
    git commit -m "run prepare_release.sh"

#    echo "(4/4) [GIT] Merging Changes" && \
#    git checkout dev && \
#    git merge xdev-prepare
