from argparse import Namespace
from collections import defaultdict

import torch
import multiprocessing
import os
import tensorflow as tf

ROOT_DIR = os.path.dirname(__file__)  # folder containing this file
args = Namespace(

    # Model

    model_name=dict(resnet_bands='resnet18', resnet_ms='resnet18', resnet_build='resnet18', Mlp='mlp'),
    self_attn='multihead',  # choices : [vanilla, intersample , multihead]
    hs_weight_init='random',  # [same, samescaled,random]
    model_init=['imagenet', 'imagenet','imagenet', None],
    imagenet_weight_path='/atlas/group/model_weights/imagenet_resnet18_tensorpack.npz',

    # Training

    lr_decay=0.96,
    batch_size=32,
    gpu=-1,
    max_epochs=200
    ,
    epoch_thresh=150,
    patience=20,

    lr=.0001,  # lr0001         #0.0001 nl,ms                     #    {1 × 10−5, 3 x 10-5 , 10-4 }
    fc_reg=0,                                               #{ 0 , 1, 10-3 , 10-2}
    # fc01_conv01_lr0001        fc001_conv001_lr0001       fc001_conv001_lr001   fc001_conv001_lr01       fc01_conv01_lr001
    conv_reg=0,

    # data

    data_path='/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_tfrecords',

    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_large',
    #'/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_tfrecords',
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_large',
    #'/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_tfrecords',
    buildings_records='/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_buildings_large',
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_buildings_large',
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_buildings_large',
    #'/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_buildings',

    # '/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_raw_buildings_large/'
    label_name='wealthpooled',  # urban_rural
    cache=['train', 'train_eval', 'val'],
    augment=True,
    clipn=True,
    normalize='DHS',
    dataset='DHS_OOC',            #Features, #Wilds
    fold='B',
    ls_bands= None,
    nl_band=None,  # [None , merge , split]
    nl_label=None,  # [center, mean,None]
    include_buildings=True,
    scaler_features_keys={'urban_rural': tf.float32},
    metadata=None,
    #['urban_rural', 'country'],
    # ['locs'],

    # keep_frac {keep_frac}

    # Experiment

    seed=123,
    experiment_name='DHS_OOC_B_buildings_larger_PE150',
    out_dir=os.path.join(ROOT_DIR, 'outputs', 'dhs_ooc'),
    init_ckpt_dir=None,
    group=  None,
    # 'urban',

    loss_type='regression',
    lamda=0.5,
    num_outputs=1,
    resume=None,
    weight_model=None,
    # '/atlas/u/amna/rural-urban-Modelbias/outputs/dhs_ooc/DHS_OOC_A_nl_random_b32_fc01_conv01_lr0001/best.ckpt',
    accumlation_steps=1,
    metric=['r2'],

    # Visualization
    # wandb project:
    wandb_p="test-project",
    entity="bias_migitation",

)
args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
args.gpus = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args.bands_channels = {'rgb': 3, 'ms': 7, 'split': 2, 'merge': 1, 'locs': 2, 'country': 1,'urban_rural':1 ,'buildings': 1}
'''
if not args.include_buildings:
    args.in_channels = args.bands_channels.get(args.ls_bands, 0) + args.bands_channels.get(args.nl_band, 0)
else:
    args.in_channels = args.bands_channels.get(args.ls_bands, 0) + args.bands_channels.get(args.nl_band, 0) + 1

if args.input == 'locs':
    args.in_channels = 2
'''

args.in_channels = [1]
'''
if args.ls_bands:
  args.in_channels.append(args.bands_channels.get(args.ls_bands, 0))
if args.nl_band:
  args.in_channels.append(args.bands_channels.get(args.nl_band, 0) )
if args.include_buildings:
    args.in_channels.append(args.bands_channels['buildings'])
if args.metadata:
    args.in_channels.append(args.bands_channels[args.metadata[0]]+args.bands_channels[args.metadata[1]])
'''