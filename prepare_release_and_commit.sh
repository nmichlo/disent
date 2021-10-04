#!/bin/bash

echo "(1/3) [GIT] Creating Prepare Branch" && \
    git checkout -b xdev-prepare && \
    \
    echo "(2/3) [PREPARE]" && \
    ./prepare_release.sh && \
    \
    echo "(3/3) [GIT] Committing Files" && \
    git add .  && \
    git commit -m "run prepare_release.sh"

#    echo "(4/4) [GIT] Merging Changes" && \
#    git checkout dev && \
#    git merge xdev-prepare
