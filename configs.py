from argparse import Namespace
import torch
import multiprocessing
import os


ROOT_DIR = os.path.dirname(__file__)  # folder containing this file
args = Namespace(

    # Model

    model_name='resnet18',
    hs_weight_init='samescaled',       #[same, samescaled,random]
    model_init=None,
    imagenet_weight_path= '/atlas/group/model_weights/imagenet_resnet18_tensorpack.npz',

    # Training

    lr_decay=0.96,
    batch_size=32,
    gpu=-1,
    max_epochs=200,

    lr=.0001,
    fc_reg=.001,                #fc01_conv01_lr0001        fc001_conv001_lr0001       fc001_conv001_lr001   fc001_conv001_lr01       fc01_conv01_lr001
    conv_reg=.001,

    # data

    data_path='/atlas/u/erikrozi/bias_mitigation/africa_poverty_clean/data/dhs_tfrecords',
    label_name='wealthpooled',
    cache=['train', 'train_eval', 'val'],
    augment=True,
    clipn=True,
    ooc=True,
    dataset='DHS_OOC',
    fold='C',
    ls_bands='ms',
    nl_band=None,  # [None , merge , split]
    nl_label=None,  # [center, mean,None]
    scaler_features_keys= None    ,#{'urban_rural':tf.float32},
    # keep_frac {keep_frac}

    # Experiment

    seed=123,
    experiment_name='DHS_OOC_C_ms_samescaled',
    out_dir=os.path.join(ROOT_DIR, 'outputs'),
    init_ckpt_dir=None,
    group=None,

    loss_type='regression',
    num_outputs=1,
    resume=None,

    #Visualization
    #wandb project:
    wandb_p='Resnet_bias'

)
args.num_workers = multiprocessing.cpu_count()
args.no_of_gpus = torch.cuda.device_count()
args.bands_channels = {'rgb': 3, 'ms': 7,'split':2, 'merge':1}  # TODO handle none key values
args.in_channels = args.bands_channels.get(args.ls_bands,0) + args.bands_channels.get(args.nl_band,0)
