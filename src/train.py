# Setup Experiment #look at utils/trainer.py
import argparse
import copy
import json
import os
import time

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger  # newline 1
from pytorch_lightning.callbacks import ModelCheckpoint
from batchers.dataset import Batcher
from batchers.torch_dataset import Data
from models.model_generator import get_model
from src.trainer import ResTrain
from utils.utils import get_paths, dotdict, init_model, parse_arguments, get_full_experiment_name
from src.configs import args as default_args
from pytorch_lightning import seed_everything

import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                    shuffle=True, num_workers=2)


data_dir = './np_data'


# ROOT_DIR = os.path.dirname(__file__)  # folder containing this file

def setup_experiment(model, train_loader, valid_loader, checkpoints, args):
    # setup lightining model params
    params = dict(model=model, lr=args.lr, weight_decay=args.conv_reg, loss_type=args.loss_type,
                  num_outputs=args.num_outputs, metric='r2')

    # setting experiment_path
    experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                          args.fc_reg, args.conv_reg, args.lr)

    dirpath = os.path.join(args.out_dir, 'dhs_ooc', experiment)
    print(f'checkpoints directory: {dirpath}')
    os.makedirs(dirpath, exist_ok=True)  # check if it can be created automatically

    # logger
    logger = TensorBoardLogger(os.path.join(args.out_dir, f"{args.model_name}_logs"), name=args.model_name)
    #logger=WandbLogger( name=args.model_name,save_dir=os.path.join(args.out_dir, f"{args.model_name}_logs"))

    # lightning model , trainer
    litmodel = ResTrain(**params)
    checkpoint_callback = ModelCheckpoint(monitor=args.monitor, mode=args.mode,
                                          filename="{resnet}-{epoch:02d}-{val_loss:.2f}",
                                          dirpath=dirpath,
                                          verbose=True,
                                          # save_top_k=1,
                                          # sanity check
                                          save_last=True)

    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.no_of_gpus,
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         resume_from_checkpoint=args.resume,
                         precision=16,
                         # overfit_batches=1,
                         #distributed_backend='ddp',
                         #strategy='ddp',
                         #num_nodes=2,
                         profiler='simple',
                         flush_logs_every_n_steps=50)
    #logger.watch(litmodel,log='all')
    # accumulate_grad_batches=16, )  # understand what it does exactly
    if checkpoints:
        print(f'Initializing using pretrained lightning model:\n{checkpoints}')
        pretrained_model = ResTrain(**params)
        pretrained_model.load_from_checkpoint(checkpoint_path=checkpoints, **params, strict=False)
        litmodel.model = copy.deepcopy(pretrained_model.model)
    # start=time.time()
    # print('before dataloader')
    # dataloader = torch.utils.data.DataLoader(Data(data_dir=data_dir), batch_size=32, num_workers=args.num_workers , pin_memory=True,
    #                                         prefetch_factor=2, shuffle=True)

    # print(f'Finished dataloader in {time.time() -start} seconds')
    # print(' values in dataloader ',(next(iter(dataloader))))
    # trainer.fit(litmodel, dataloader)
    trainer.fit(litmodel, train_loader, valid_loader)

    # trainer.test(litmodel,train_loader)

    torch.save(litmodel.model.state_dict(),
               dirpath)  # save the model itself (resnetms for example)rather than saving the lighting model

    best_model_ckpt = checkpoint_callback.best_model_path
    best_model_score = checkpoint_callback.best_model_score

    return best_model_ckpt, best_model_score, dirpath


def main(args):
    seed_everything(args.seed)
    # dataloader
    paths_train = get_paths(args.dataset, ['train'], args.fold, args.data_path)
    paths_valid = get_paths(args.dataset, ['val'], args.fold, args.data_path)

    # save data_params for later use
    data_params = dict(dataset=args.dataset, fold=args.fold, ls_bands=args.ls_bands, nl_band=args.nl_band,
                       label_name=args.label_name,
                       nl_label=args.nl_label, batch_size=args.batch_size, groupby=args.group)

    batcher_train = Batcher(paths_train[:6000], args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS', args.augment, args.clipn, args.batch_size, groupby=args.group,
                            cache=False)
    batcher_valid = Batcher(paths_valid, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS', args.augment, args.clipn, args.batch_size, groupby=args.group,
                            cache='train_evl' in args.cache)
    # Test Loop
    '''
    z=0
    for  j in batcher_train:
        if z<3:
           print('fetching data ....')
           print( j['images'] )
        z+=1
    '''

    # model
    ckpt, pretrained = init_model(args.model_init, args.init_ckpt_dir, )
    model = get_model(args.model_name, in_channels=args.in_channels, pretrained=pretrained, ckpt_path=ckpt)  ##TEST
    '''
    model.to('cuda')
    for record in batcher_train:
        start1=time.time()
        x=torch.tensor(record['images'],device='cuda')
        x = x.reshape(-1, x.shape[-1], x.shape[-3], x.shape[-2])
        start2= time.time()
        print(f'time in batch {start2 - start1}')
        output=model(x)
        print(f'time in model{time.time()-start2}')
    '''

    #best_model_ckpt, _, dirpath = setup_experiment(model, batcher_train, batcher_valid, args.checkpoints, args)
    best_model_ckpt, _, dirpath = setup_experiment(model, trainloader,trainloader ,args.checkpoints, args)
    print(f'Path to best model found during training: \n{best_model_ckpt}')

    # saving data_param:
    params_filepath = os.path.join(dirpath, 'data_params.json')
    with open(params_filepath, 'w') as config_file:
        json.dump(data_params, config_file, indent=4)

    # saving resnet model params filepath
    params = dict(model_name=args.model_name, in_channels=args.in_channels)
    params_filepath = os.path.join(dirpath, 'params.json')
    with open(params_filepath, 'w') as config_file:
        json.dump(params, config_file, indent=4)


if __name__ == "__main__":
    print('GPUS:', torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser, default_args)
    args = dotdict(args)

    main(args)
