import os
import subprocess

import time

from configs import args
from utils.utils import get_full_experiment_name

ROOT_DIR = os.path.dirname(__file__)
log_dir = os.path.join(ROOT_DIR, 'logs/DHS_OOC/')
mem = 100
command = 'python -m src.train2'
#output_path = os.path.join(log_dir, 'output.log')
#command = f'{command} >& "{output_path}" &'
experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                      args.fc_reg, args.conv_reg, args.lr)
dirpath = os.path.join(args.out_dir, experiment)
with open(os.path.join(ROOT_DIR, 'scripts', 'train_model_slurm.sh'), 'r') as template_file:
    template = template_file.read()
    full_slurm_script = template.format(
        SLURM_MEM=f'{mem}G',
        SLURM_JOB_NAME=f'slurm',
        SLURM_OUTPUT_LOG=dirpath,
        content=command)

    slurm_sh_path = os.path.join(dirpath, 'slurm.sh')
    with open(slurm_sh_path, 'w') as f:
        f.write(full_slurm_script)
    subprocess.run(['sbatch', slurm_sh_path])
