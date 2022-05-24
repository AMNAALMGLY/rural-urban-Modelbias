from argparse import Namespace
from collections import defaultdict

import torch
import multiprocessing
import os
import tensorflow as tf

ROOT_DIR = '/network/scratch/a/amna.elmustafa/rural-urban-Modelbias'
args = Namespace(

    # Model

    model_name=dict(resnet_bands='resnet18', resnet_ms='resnet18', resnet_build='resnet18', Mlp='mlp'),
    self_attn=None,  # choices : [multihead, multihead_space , multihead_uniform,multihead_random]
    hs_weight_init='random',  # [same, samescaled,random]
    model_init=['imagenet', 'imagenet','imagenet', None],
    imagenet_weight_path='./imagenet_resnet18_tensorpack.npz',
    p_size=224,
    stride=60,
    blocks=4,
    randcrop=False,            #this is for cropping in the forward pass
    rand_crop=0,               #This for cropping from the dataset specifying size    (mostly cropping size is not the same as patching size for attention)
    offset=20,

    # Training

    scheduler='warmup_cos',           #warmup_step, #warmup_cos   #step    #cos  #exp
    lr_decay=0.96,
    batch_size=64,
    gpu=-1,
    max_epochs=200,
    epoch_thresh=150,
    patience=20,
    #A Building :wd=0
    lr=.001,  # lr0001         #0.0001 nl,ms                     #    {1 × 10−5, 3 x 10-5 , 10-4 }
    fc_reg=0.00001,                                               #{ 0 , 1, 10-3 , 10-2}
    # fc01_conv01_lr0001        fc001_conv001_lr0001       fc001_conv001_lr001   fc001_conv001_lr01       fc01_conv01_lr001
    conv_reg=0.00001,



    # data
    image_size=355,
    crop=224,
    data_path='/network/scratch/a/amna.elmustafa/dhs_tfrecords_large/',
    #atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_raw_ultralarge_onlynl .
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_large',
    #'/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_tfrecords',
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_large',
    #'/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_tfrecords',
    buildings_records='/network/scratch/a/amna.elmustafa/dhs_tfrecords_buildings_large', #or None

    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_buildings_large',
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_buildings_large',
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_buildings_large',
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_buildings_large',
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_buildings_large',
    #'/atlas/u/erikrozi/bias_mitigation/dhs_tfrecords_buildings_large',
    #'/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_buildings',
    labels_path='/network/scratch/a/amna.elmustafa/rural-urban-Modelbias/preprocessing/dhs_labels_quantile.csv',
    #dhs_labels_quantile_sqrt.csv',
    #dhs_labels_quantile.csv',
    #dhs_labels_std.csv',
    #dhs_labels_normalized.csv',
    #'/network/scratch/a/amna.elmustafa/rural-urban-Modelbias/preprocessing/dhs_labels_processed.csv',

    label_name='water_index',  #['sanitation_index , women_edu , women_bmi']
    cache=['train', 'train_eval', 'val'],
    augment=True,
    clipn=True,
    normalize='DHS',
    dataset='DHS_OOC',            #Features, #Wilds

    fold='E',
    ls_bands=None,
    nl_band=None,  # [None , merge , split]
    nl_label=None,  # [center, mean,None]
    include_buildings=True,  # True or false    #True goes with buildings_records , False goes with building_records None
    scaler_features_keys={'urban_rural': tf.float32},
    metadata=None,
    #['urban_rural', 'country'],
    # ['locs'],

    # keep_frac {keep_frac}

    # Experiment
    seed=123,
    experiment_name='DHS_OOC_build_resnet_waterqntLoss', #change (water ) to  current label name and (build) to current input band
    #'DHS_OOC_build_resnet_attn224_sanitation',
    out_dir=os.path.join(ROOT_DIR, 'outputs', 'dhs_ooc','ablation_study'),
    init_ckpt_dir='/network/scratch/a/amna.elmustafa/outputs/dhs_ooc/ablation_study/DHS_OOC_NL_resnet_water_qnt_b128_fce-05_conve-05_lr001_crop224_foldA/best.ckpt',
    #DHS_OOC_A_building_no_attn_pretrained_attn_355P224_b32_fce-05_conve-05_lr0001_crop224_foldA/best.ckpt',
    #'/network/scratch/a/amna.elmustafa/outputs/dhs_ooc/ablation_study/DHS_OOC_D_NL_no_attn_355P100_b32_fce-05_conve-05_lr0001_crop100/best.ckpt',
    #'/network/scratch/a/amna.elmustafa/outputs/dhs_ooc/ablation_study/DHS_OOC_C_building_no_attn_355P100_b32_fce-05_conve-05_lr0001_crop100/best.ckpt',
    #'/network/scratch/a/amna.elmustafa/outputs/dhs_ooc/ablation_study/DHS_OOC_D_builiding_no_attn_355P100_b32_fc01_conv01_lr0001_crop100/best.ckpt',
    #'/network/scratch/a/amna.elmustafa/outputs/dhs_ooc/ablation_study/DHS_OOC_A_builiding_no_attn_355P100_b32_fc01_conv01_lr0001_crop100/best.ckpt',
    #'/network/scratch/a/amna.elmustafa/outputs/dhs_ooc/ablation_study/DHS_OOC_E_building_no_attn_355P100_b32_fce-05_conve-05_lr0001_crop100/best.ckpt',
    group=  None,

    # 'urban',

    loss_type='quantile',
    #'l1',
    num_quantiles=3,
    lamda=0.5,
    num_outputs=1,
    resume=None,
    weight_model=None,
    # '/atlas/u/amna/rural-urban-Modelbias/outputs/dhs_ooc/DHS_OOC_A_nl_random_b32_fc01_conv01_lr0001/best.ckpt',
    accumlation_steps=2,
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
