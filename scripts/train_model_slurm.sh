#!/bin/bash

##########################
# This file is a template. Fill in the blanks with python.
#
# To stop this job, get the jobid using
#   squeue -u <username>
# then cancel the job with
#   scancel <jobid>
##########################
export CUDA_VISIBLE_DEVICES=0,1

# choose the machine
SBATCH --partition=atlas --exclude=atlas6,atlas20

# set the machine parameters
NO_GPUS=1
SBATCH --nodes=1 --cpus-per-task=10 --mem={SLURM_MEM} --gres=gpu:${NO_GPUS}

# set the job name
SBATCH --job-name={SLURM_JOB_NAME}

# set maximum time for job to run
# indefinite job: --time=0
# days/hours: --time=days-hours
SBATCH --time=2-0

# set the output log name
SBATCH --output={SLURM_OUTPUT_LOG}/%j.out

SBATCH --wrap={content}

SBATCH --error={SLURM_OUTPUT_LOG}/%j.err
# print out Slurm Environment Variables
echo "
Slurm Environment Variables:
- SLURM_JOBID=$SLURM_JOBID
- SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST
- SLURM_NNODES=$SLURM_NNODES
- SLURMTMPDIR=$SLURMTMPDIR
- SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR

"

# slurm doesn't source .bashrc automatically
source ~/.bashrc

project_dir="/atlas/u/amna/rural-urban-Modelbias/"
echo "Setting directory to: $project_dir"
cd $project_dir

# list out some useful information
echo "
Basic system information:
- Date: $(date)
- Hostname: $(hostname)
- User: $USER
- pwd: $(pwd)
"

conda activate envi

{content}

echo "All jobs launched!"
echo "Waiting for child processes to finish..."
wait
echo "Done!"
