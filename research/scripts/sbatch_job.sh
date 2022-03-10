#!/bin/bash
#SBATCH --nodes=1 --ntasks=1

# ========================================================================= #
# CHECKS & SETUP                                                            #
# ========================================================================= #

# make sure that this was launched as an array
# make sure that we have an array file specified
if [ -z "$SLURM_ARRAY_JOB_ID" ];  then echo "SLURM_ARRAY_JOB_ID is not defined... Are you sure the job was launched as an array?" ;  exit 1 ; fi
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then echo "SLURM_ARRAY_TASK_ID is not defined... Are you sure the job was launched as an array?" ; exit 1 ; fi
if [ -z "$ARGS_FILE" ];           then echo "ARGS_FILE is not defined... Cannot obtain job settings!" ;                              exit 1 ; fi
if [ -z "$SCRIPT_DIR" ];          then echo "SCRIPT_DIR is not set... Cannot find helper.sh";                                        exit 1; fi
echo "[RUNNING JOB]: $SLURM_ARRAY_TASK_ID"

# setup the python environment
. "$HOME/installed/pyenv/versions/miniconda3-latest/etc/profile.d/conda.sh"
conda activate disent-conda

# save the working directory
_run_pwd="$(pwd)"
_run_dir="$_run_pwd/$SLURM_ARRAY_JOB_ID/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

# load `local_run`
. "$SCRIPT_DIR/helper.sh"

# set the working directory
mkdir -p "$_run_dir"
cd "$_run_dir" || exit 1
echo "- set working directory: $(pwd)"
echo

# ========================================================================= #
# RUN                                                                       #
# ========================================================================= #

# get the line in the file corresponding to the array number
# - sed starts indexing at 1, not zero
LINE=$(sed -n "$SLURM_ARRAY_TASK_ID"p "$ARGS_FILE")

# run experiment with the specified arguments
local_run $LINE
