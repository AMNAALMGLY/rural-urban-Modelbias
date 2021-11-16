# Setup Experiment #look at utils/trainer.py
import argparse
import copy
import json
import os
import torch
from batchers.dataset import Batcher
from models.model_generator import get_model
from src.trainer2 import Trainer
from utils.utils import get_paths, dotdict, init_model, parse_arguments, get_full_experiment_name,load_from_checkpoint
from configs import args as default_args
from pytorch_lightning import seed_everything
import wandb
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
#






# ROOT_DIR = os.path.dirname(__file__)  # folder containing this file

def setup_experiment(model, train_loader, valid_loader, resume_checkpoints, args):

    # if resume training:
    if resume_checkpoints:
        print((f'resuming training from {resume_checkpoints}'))
        model = load_from_checkpoint(resume_checkpoints, model)
    # setup lightining model params
    params = dict(model=model, lr=args.lr, weight_decay=args.conv_reg, loss_type=args.loss_type,
                  num_outputs=args.num_outputs, metric='r2')

    wandb.config.update(params)
    # setting experiment_path
    experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                          args.fc_reg, args.conv_reg, args.lr)

    dirpath = os.path.join(args.out_dir, 'dhs_ooc', experiment)
    print(f'checkpoints directory: {dirpath}')
    os.makedirs(dirpath, exist_ok=True)  # check if it can be created automatically



    # lightning model , trainer
    trainer = Trainer(save_dir=dirpath,**params)


    best_loss, path= trainer.fit( train_loader, valid_loader,max_epochs=args.max_epochs,gpus='cuda')


    return best_loss, path, dirpath


def main(args):
    seed_everything(args.seed)
    # dataloader
    paths_train = get_paths(args.dataset, ['train'], args.fold, args.data_path)
    paths_valid = get_paths(args.dataset, ['val'], args.fold, args.data_path)

    # save data_params for later use
    data_params = dict(dataset=args.dataset, fold=args.fold, ls_bands=args.ls_bands, nl_band=args.nl_band,
                       label_name=args.label_name,
                       nl_label=args.nl_label, batch_size=args.batch_size, groupby=args.group)
    wandb.config.update(data_params)
    batcher_train = Batcher(paths_train, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS', args.augment, args.clipn, args.batch_size, groupby=args.group,
                            cache=True)
    batcher_valid = Batcher(paths_valid, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS', args.augment, args.clipn, args.batch_size, groupby=args.group,
                            cache=True)

    ckpt, pretrained = init_model(args.model_init, args.init_ckpt_dir, )
    model = get_model(args.model_name, in_channels=args.in_channels, pretrained=pretrained, ckpt_path=ckpt)  ##TEST


    best_loss,path, dirpath , = setup_experiment(model, batcher_train, batcher_valid, args.checkpoints, args)

    print(f'Path to best model found during training: \n{path}')

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
    wandb.init(project="rual-urban-torch", config={})
    print('GPUS:', torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser, default_args)
    args = dotdict(args)

    main(args)
