# Setup Experiment #look at utils/trainer.py
import argparse
import copy
import json
import os
import torch
from torch import nn

from batchers.dataset import Batcher
from models.model_generator import get_model
from src.trainer2 import Trainer
from utils.utils import get_paths, dotdict, init_model, parse_arguments, get_full_experiment_name,load_from_checkpoint
from configs import args as default_args
from utils.utils import seed_everything
import wandb
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
#






# ROOT_DIR = os.path.dirname(__file__)  # folder containing this file

def setup_experiment(model, train_loader, valid_loader, resume_checkpoints, args):


    # if resume training:
    if resume_checkpoints:
        print((f'resuming training from {resume_checkpoints}'))
        # redefine the model according to num_outputs              #TODO refactor this model redefinintion to be less redundant
        fc = nn.Linear(model.fc.in_features, args.num_outputs)
        model.fc = fc
        model = load_from_checkpoint(resume_checkpoints, model)

    # setup Trainer params
    params = dict(model=model, lr=args.lr, weight_decay=args.conv_reg, loss_type=args.loss_type,
                  num_outputs=args.num_outputs, metric='r2')

    wandb.config.update(params)
    # setting experiment_path
    experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                          args.fc_reg, args.conv_reg, args.lr)

    dirpath = os.path.join(args.out_dir, 'dhs_ooc', experiment)
    print(f'checkpoints directory: {dirpath}')
    os.makedirs(dirpath, exist_ok=True)


    # Trainer
    trainer = Trainer(save_dir=dirpath,**params)


    best_loss, path= trainer.fit( train_loader, valid_loader,max_epochs=args.max_epochs,gpus='cuda')


    return best_loss, path,


def main(args):
    seed_everything(args.seed)

    # setting experiment_path
    experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                          args.fc_reg, args.conv_reg, args.lr)



    dirpath = os.path.join(args.out_dir, 'dhs_ooc', experiment)

    os.makedirs(dirpath, exist_ok=True)

    # save data_params for later use
    data_params = dict(dataset=args.dataset, fold=args.fold, ls_bands=args.ls_bands, nl_band=args.nl_band,
                       label_name=args.label_name,
                       nl_label=args.nl_label, batch_size=args.batch_size, groupby=args.group)

    params_filepath = os.path.join(dirpath, 'data_params.json')
    with open(params_filepath, 'w') as config_file:
        json.dump(data_params, config_file, indent=4)

    # saving resnet model params filepath
    params = dict(model_name=args.model_name, in_channels=args.in_channels)
    params_filepath = os.path.join(dirpath, 'params.json')
    with open(params_filepath, 'w') as config_file:
        json.dump(params, config_file, indent=4)

    wandb.config.update(data_params)

    # dataloader
    paths_train = get_paths(args.dataset, ['train'], args.fold, args.data_path)

    paths_valid = get_paths(args.dataset, ['val'], args.fold, args.data_path)

    paths_test=get_paths(args.dataset, ['test'], args.fold, args.data_path)

    batcher_train = Batcher(paths_train, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS',args.augment, args.clipn, args.batch_size, groupby=args.group,
                            cache=True,shuffle=True)

    batcher_valid = Batcher(paths_valid, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS', False, args.clipn, args.batch_size, groupby=args.group,
                            cache=True,shuffle=False)

    batcher_test=Batcher(paths_test[:1000], args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS', False, args.clipn, args.batch_size, groupby=args.group,
                            cache=True,shuffle=False)

    ckpt, pretrained = init_model(args.model_init, args.init_ckpt_dir, )

    model = get_model(args.model_name, in_channels=args.in_channels, pretrained=pretrained, ckpt_path=ckpt)


    best_loss,best_path,_ = setup_experiment(model, batcher_train, batcher_valid, args.resume, args)

    print(f'Path to best model found during training: \n{best_path}')


#TODO save hyperparameters .




if __name__ == "__main__":
    wandb.init(project="Resnet_bias", config={})
    print('GPUS:', torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser, default_args)
    args = dotdict(args)

    main(args)
