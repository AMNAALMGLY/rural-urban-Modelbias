from argparse import Namespace
import torch
import multiprocessing
import os

ROOT_DIR = os.path.dirname(__file__)  # folder containing this file

args = Namespace(

    # Model

    model_name='resnet18',
    hs_weight_init='same',       #[same, samescaled,random]
    model_init='imagenet',

    # Training

    lr_decay=0.96,
    batch_size=128,
    gpu=-1,
    max_epochs=100,

    lr=3*0.001,
    fc_reg=0.001,
    conv_reg=0.001,

    # data

    data_path='/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_tfrecords',
    label_name='wealthpooled',
    cache=['train', 'train_eval', 'val'],
    augment=True,
    clipn=True,
    ooc=True,
    dataset='DHS_OOC',
    fold='A',
    ls_bands='rgb',
    nl_band=None,  # [None , merge , split]
    nl_label=None,  # [center, mean,None]
    scaler_features_keys=None,
    # keep_frac {keep_frac}

    # Experiment

    monitor='train_loss',
    mode='min',
    seed=123,
    experiment_name='DHS_OOC_A_ms_same',
    out_dir=os.path.join(ROOT_DIR, 'outputs/'),
    ckpt=None,
    group=None,
    loss_type='classification',
    num_outputs=10,
    #resume='../last.ckpt',
    resume=None,
    checkpoints= None,
)
args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
args.bands_channels = {'rgb': 3, 'ms': 7}  # TODO handle none key values
args.in_channels = args.bands_channels[args.ls_bands] + 1 if args.nl_band else args.bands_channels[args.ls_bands]
