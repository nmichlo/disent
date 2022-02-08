#!/bin/bash

# ========================================================================= #
# Description                                                               #
# ========================================================================= #

# before sourcing this script, it requires the following variables to be exported:
# - PROJECT: str
# - PARTITION: str
# - PARALLELISM: int

# source this script from the script you use to run the experiment
# 1. gets and exports the path to the root
# 2. changes the working directory to the root
# 3. exports a helper function that runs the script in the background, with
#    the correct python path and settings

if [ -z "$PROJECT" ]; then echo "PROJECT is not set"; exit 1; fi
if [ -z "$PARTITION" ]; then echo "PARTITION is not set"; exit 1; fi
if [ -z "$PARALLELISM" ]; then echo "PARALLELISM is not set"; exit 1; fi
if [ -z "$USERNAME" ]; then echo "USERNAME is not set"; exit 1; fi
if [ -z "$PY_RUN_FILE" ]; then PY_RUN_FILE='experiment/run.py'; fi

export PY_RUN_FILE

# ========================================================================= #
# Helper                                                                    #
# ========================================================================= #

# get the root directory
SCRIPT_DIR=$(dirname "$(realpath -s "$0")")
ROOT_DIR="$(realpath -s "$SCRIPT_DIR/../../..")"

# cd into the root, exit on failure
cd "$ROOT_DIR" || exit 1
echo "working directory is: $(pwd)"

# hydra search path and plugins todo: make this configurable?
_SEARCH_PATH="${ROOT_DIR}/research/config"

function submit_sweep() {
    echo "SUBMITTING SWEEP:" "$@"
    PYTHONPATH="$ROOT_DIR" DISENT_CONFIGS_PREPEND="$_SEARCH_PATH" python3 "$PY_RUN_FILE" \
        -m \
        run_launcher=slurm \
        dsettings.launcher.partition="$PARTITION" \
        settings.job.project="$PROJECT" \
        settings.job.user="$USERNAME" \
        hydra.launcher.array_parallelism="$PARALLELISM" \
        "$@" \
        & # run in background
}

function local_run() {
    echo "RUNNING:" "$@"
    PYTHONPATH="$ROOT_DIR" DISENT_CONFIGS_PREPEND="$_SEARCH_PATH" python3 "$PY_RUN_FILE" \
        run_launcher=local \
        settings.job.project="$PROJECT" \
        settings.job.user="$USERNAME" \
        "$@"
}

function local_sweep() {
    echo "RUNNING SWEEP:" "$@"
    PYTHONPATH="$ROOT_DIR" DISENT_CONFIGS_PREPEND="$_SEARCH_PATH" python3 "$PY_RUN_FILE" \
        -m \
        run_launcher=local \
        settings.job.project="$PROJECT" \
        settings.job.user="$USERNAME" \
        "$@"
}

# export
export ROOT_DIR
export submit_sweep
export local_run

# debug hydra
HYDRA_FULL_ERROR=1
export HYDRA_FULL_ERROR

# ========================================================================= #
# Slurm Helper                                                              #
# ========================================================================= #


function num_idle_nodes() {
  if [ -z "$1" ]; then echo "partition (first arg) is not set"; exit 1; fi
  # number of idle nodes
  num=$(sinfo --partition="$1" --noheader -O Nodes,Available,StateCompact | awk '{if($2 == "up" && $3 == "idle"){print $1}}')
  if [ -z "$num" ]; then num=0; fi
  echo $num
}

function clog_cudaless_nodes() {
  if [ -z "$1" ]; then echo "partition is not set"; exit 1; fi
  if [ -z "$2" ]; then echo wait=120; else wait="$2"; fi
  if [ -z "$3" ]; then echo name="NO-CUDA"; else name="$3"; fi
  # clog idle nodes
  n=$(num_idle_nodes "$1")
  if [ "$n" -lt "1" ]; then
    echo -e "\e[93mclogging skipped! no idle nodes found on partition '$1'\e[0m";
  else
    echo -e "\e[92mclogging $n nodes on partition '$1' for ${wait}s if cuda is not available!\e[0m";
    sbatch --array=1-"$n" --partition="$1" --job-name="$name" --output=/dev/null --error=/dev/null \
           --wrap='python -c "import torch; import time; cuda=torch.cuda.is_available(); print(\"CUDA:\", cuda, flush=True); print(flush=True); time.sleep(5 if cuda else '"$wait"');"'
  fi
}

function clog_cuda_nodes() {
  if [ -z "$1" ]; then echo "partition is not set"; exit 1; fi
  if [ -z "$2" ]; then echo wait=120; else wait="$2"; fi
  if [ -z "$3" ]; then echo name="HAS-CUDA"; else name="$3"; fi
  # clog idle nodes
  n=$(num_idle_nodes "$1")
  if [ "$n" -lt "1" ]; then
    echo -e "\e[93mclogging skipped! no idle nodes found on partition '$1'\e[0m";
  else
    echo -e "\e[92mclogging $n nodes on partition '$1' for ${wait}s if cuda is available!\e[0m";
    sbatch --array=1-"$n" --partition="$1" --job-name="$name" --output=/dev/null --error=/dev/null \
           --wrap='python -c "import torch; import time; cuda=torch.cuda.is_available(); print(\"CUDA:\", cuda, flush=True); print(flush=True); time.sleep(5 if not cuda else '"$wait"');"'
  fi
}

export num_idle_nodes
export clog_cudaless_nodes
export clog_cuda_nodes

# ========================================================================= #
# End                                                                       #
# ========================================================================= #
