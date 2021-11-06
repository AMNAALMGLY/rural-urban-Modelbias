from argparse import Namespace
import torch
import multiprocessing
import os
ROOT_DIR = os.path.dirname(__file__)  # folder containing this file

args = Namespace(

    # Model

    model_name='resnet18',
    hs_weight_init='same',

    # Training

    lr_decay=0.96,
    batch_size=64,
    gpu=0,
    max_epochs=150,

    lr=0.001,
    fc_reg=0.001,
    conv_reg=0.001,

    # data

    label_name='wealthpooled',
    cache=['train', 'train_eval', 'val'],
    augment=True,
    ooc=True,
    dataset='DHS_OOC',
    fold='A',
    ls_bands='ms',
    nl_band=None,     #[None , merge , split]
    nl_label='None',     #[center, mean]
    # keep_frac {keep_frac}

    # Experiment

    seed=123,
    experiment_name='new_experiment',
    out_dir=os.path.join(ROOT_DIR, 'outputs/'),
    ckpt=None

)
args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
