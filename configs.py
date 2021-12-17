from argparse import Namespace
import torch
import multiprocessing
import os

ROOT_DIR = os.path.dirname(__file__)  # folder containing this file
args = Namespace(

    # Model

    model_name='resnet18',
    hs_weight_init='random',  # [same, samescaled,random]
    model_init='imagenet',
    imagenet_weight_path='/atlas/group/model_weights/imagenet_resnet18_tensorpack.npz',

    # Training

    lr_decay=0.96,
    batch_size=90,
    gpu=-1,
    max_epochs=200,
    epoch_thresh=150,
    patience=20,

    lr=.0001,  # lr0001         #0.0001 nl,ms
    fc_reg=0.01,
    # fc01_conv01_lr0001        fc001_conv001_lr0001       fc001_conv001_lr001   fc001_conv001_lr01       fc01_conv01_lr001
    conv_reg=0.01,

    # data

    data_path='/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_tfrecords',
    buildings_records='/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_buildings',
    label_name='wealthpooled', #urban_rural
    cache=['train', 'train_eval', 'val'],
    augment=True,
    clipn=True,
    ooc=True,
    dataset='DHS_OOC',
    fold='A',
    ls_bands=None,
    nl_band='split',  # [None , merge , split]
    nl_label=None,  # [center, mean,None]
    include_buildings=True,
    scaler_features_keys=None,  # {'urban_rural':tf.float32},

    # keep_frac {keep_frac}

    # Experiment

    seed=123,
    experiment_name='DHS_OOC_A_nl_custom',
    out_dir=os.path.join(ROOT_DIR, 'outputs', 'dhs_ooc'),
    init_ckpt_dir=None,
    group=None,

    loss_type='custom',
    num_outputs=1,
    resume=None,
    weight_model=None,
    #'/atlas/u/amna/rural-urban-Modelbias/outputs/dhs_ooc/DHS_OOC_A_nl_random_b32_fc01_conv01_lr0001/best.ckpt',
    accumlation_steps=1,

    # Visualization
    # wandb project:
    wandb_p="test-project",
    entity="bias_migitation",

)
args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
args.bands_channels = {'rgb': 3, 'ms': 7, 'split': 2, 'merge': 1}
if not args.include_buildings:
    args.in_channels = args.bands_channels.get(args.ls_bands, 0) + args.bands_channels.get(args.nl_band, 0)
else:
    args.in_channels = args.bands_channels.get(args.ls_bands, 0) + args.bands_channels.get(args.nl_band, 0) + 1
