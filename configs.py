from argparse import Namespace
import torch
import multiprocessing
import os
import  tensorflow as tf

ROOT_DIR = os.path.dirname(__file__)  # folder containing this file
args = Namespace(

    # Model

    model_name='resnet18',
    hs_weight_init='samescaled',       #[same, samescaled,random]
    model_init='imagenet',

    # Training

    lr_decay=0.96,
    batch_size=64,
    gpu=-1,
    max_epochs=200,

    lr=.001,
    fc_reg=.01,
    conv_reg=.01,

    # data

    data_path='/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_tfrecords',
    label_name='wealthpooled',
    cache=['train', 'train_eval', 'val'],
    augment=True,
    clipn=True,
    ooc=True,
    dataset='DHS_OOC',
    fold='E',
    ls_bands='ms',
    nl_band=None,  # [None , merge , split]
    nl_label=None,  # [center, mean,None]
    scaler_features_keys={'urban_rural':tf.float32},
    # keep_frac {keep_frac}

    # Experiment

    seed=123,
    experiment_name='DHS_OOC_E_ms_samescaled',
    out_dir=os.path.join(ROOT_DIR, 'outputs'),
    init_ckpt_dir=None,
    group='urban',

    loss_type='regression',
    num_outputs=1,
    resume=None,

)
args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
args.bands_channels = {'rgb': 3, 'ms': 7}  # TODO handle none key values
args.in_channels = args.bands_channels[args.ls_bands] + 1 if args.nl_band else args.bands_channels[args.ls_bands]
