#!/bin/bash

##########################
# This file is a template. Fill in the blanks with python.
#
# To stop this job, get the jobid using
#   squeue -u <username>
# then cancel the job with
#   scancel <jobid>
##########################
#source you virtualenv
#cd /sailhome/amna/anaconda3
GPUS=1
echo "Number of GPUs: "${GPUS}
WRAP="python -m models.extract_features"
JOBNAME="extract_features"
LOG_FOLDER="/atlas/u/amna/logs/resnet18_logs/"
echo ${WRAP}
echo "Log Folder:"${LOG_FOLDER}
mkdir -p ${LOG_FOLDER}
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

#{content}

export CUDA_VISIBLE_DEVICES=0

sbatch --output=${LOG_FOLDER}/%j.out --error=${LOG_FOLDER}/%j.err \
    --nodes=1 --ntasks-per-node=1 --time=1-00:00:00 --mem=100G \
    --partition=atlas --cpus-per-task=10 --exclude=atlas6,atlas20,atlas22,atlas23,atlas24\
    --gres=gpu:${GPUS} --job-name=${JOBNAME} --wrap="${WRAP}"


echo "All jobs launched!"
echo "Waiting for child processes to finish..."
wait
echo "Done!"

