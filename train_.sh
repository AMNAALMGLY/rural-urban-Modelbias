#!/bin/bash
#SBATCH --job-name=ebird
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -o /home/mila/a/amna.elmustafa/sbatch_logs/%j.out
#SBATCH -e /home/mila/a/amna.elmustafa/sbatch_logs/%j.err
#SBATCH -p long


###module load anaconda/3 >/dev/null 2>&1
###. "$CONDA_ACTIVATE"
source activate envi

export CUDA_VISIBLE_DEVICES=0,1
export WANDB_API_KEY='f28ff8db512f61943604cf2be4d356bb738fc8ee'
export WANDB_ENTITY='bias_migitation'
export WANDB_PROJECT='test-project'
echo "Starting job"
python -m src.train2 $@
echo 'done'
