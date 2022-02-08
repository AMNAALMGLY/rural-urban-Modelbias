# Setup Experiment #look at utils/trainer.py
import argparse
import copy
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import nn

from batchers import dataset_constants_buildings
from batchers.dataset import Batcher

from models.model_generator import get_model, Encoder, MultiHeadedAttention, geoAttention
from src.trainer2 import Trainer
from utils.utils import get_paths, dotdict, init_model, parse_arguments, get_full_experiment_name, load_from_checkpoint, \
    load_npz
from configs import args as default_args
from utils.utils import seed_everything
import tensorflow as tf
import wandb
from torch.utils.tensorboard import SummaryWriter
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from ray import tune
import torchvision.transforms as transforms

wandp = default_args.wandb_p
entity = default_args.entity


def geo_attn_exp(model_name, MODEL_DIRS, ):
    COUNTRIES = dataset_constants_buildings.DHS_COUNTRIES
    df = pd.read_csv('data/dhs_clusters.csv', float_precision='high', index_col=False)
    df = df[df['country'].isin(COUNTRIES)].reset_index(drop=True)
    labels = df['wealthpooled'].to_numpy(dtype=np.float32)

    # Load features from pretrained Model
    f = args.fold
    model_fold_name = f'{model_name}_{f}'
    model_dir = MODEL_DIRS[model_fold_name]
    npz_path = os.path.join('outputs', 'dhs_ooc', model_dir, 'features.npz')
    npz = load_npz(npz_path, check={'labels': labels})
    features = torch.tensor(npz['features'])
    labels = torch.tensor(labels)
    print('features,labels', features.shape, labels.shape)
    # Sorting idx according to lat, lon
    countries_train_idx = (
        df[df['country'].isin(dataset_constants_buildings.SURVEY_NAMES[f'DHS_OOC_{f}']['train'])].index).to_numpy()
    countries_valid_idx = (
        df[df['country'].isin(dataset_constants_buildings.SURVEY_NAMES[f'DHS_OOC_{f}']['val'])].index).to_numpy()
    countries_test_idx = (
        df[df['country'].isin(dataset_constants_buildings.SURVEY_NAMES[f'DHS_OOC_{f}']['test'])].index).to_numpy()

    # idx = np.array(range(num_examples))

    sorted_idx = (df.sort_values(['lat', 'lon']).index).to_numpy()
    print('sorted_idx', sorted_idx)
    train_idx = sorted_idx[np.isin(sorted_idx, countries_train_idx)]
    print('train_idx', len(train_idx))
    valid_idx = sorted_idx[np.isin(sorted_idx, countries_valid_idx)]
    test_idx = sorted_idx[np.isin(sorted_idx, countries_test_idx)]

    # Dataloader
    # print('validate sorting',features[:5],features[train_idx][:5])
    train, valid, test = torch.utils.data.TensorDataset(features[train_idx],
                                                        labels[train_idx]), torch.utils.data.TensorDataset(
        features[valid_idx], labels[valid_idx]), torch.utils.data.TensorDataset(features[test_idx], labels[test_idx])

    # train, test, valid = dataset[train_idx], dataset[test_idx], dataset[valid_idx]
    trainloader, validloader, testloader = torch.utils.data.DataLoader(train,
                                                                       batch_size=64), torch.utils.data.DataLoader(
        valid, batch_size=64), torch.utils.data.DataLoader(test, batch_size=64)

    attn_layer = geoAttention()
    best_loss, path, score = setup_experiment(attn_layer, trainloader, validloader, None, args, testloader)
    return best_loss, path, score


def building_exp():
    # Train data
    dir = 'outputs/dhs_ooc/DHS_OOC_A_patches_attnLayer_b64_fc0001_conv0001_lr0001/building_sum'
    npz = load_npz(dir)
    data = npz['building_sum']
    data.unsqueeze_(-1)
    # Valid data
    dir = 'outputs/dhs_ooc/DHS_OOC_A_patches_attnLayer_b64_fc0001_conv0001_lr0001/building_sum_valid'
    npz = load_npz(dir)
    valid = npz['building_sum']
    valid.unsqueeze_(-1)

    model = get_model('mlp', 1)
    COUNTRIES = dataset_constants_buildings.DHS_COUNTRIES
    df = pd.read_csv('data/dhs_clusters.csv', float_precision='high', index_col=False)
    df = df[df['country'].isin(COUNTRIES)].reset_index(drop=True)
    labels = df['wealthpooled'].to_numpy(dtype=np.float32)
    labels = torch.tensor(labels)

    idx = np.array(range(len(labels)))
    train_idx = (
        df[df['country'].isin(dataset_constants_buildings.SURVEY_NAMES[f'DHS_OOC_{f}']['train'])].index).to_numpy()
    valid_idx = (
        df[df['country'].isin(dataset_constants_buildings.SURVEY_NAMES[f'DHS_OOC_{f}']['val'])].index).to_numpy()
    # train_idx,valid_idx=idx[np.isin(idx,countries_train_idx)]
    train, valid = torch.utils.data.TensorDataset(data, labels[train_idx]), torch.utils.data.TensorDataset(valid,
                                                                                                           labels[
                                                                                                               valid_idx])
    trainloader, validloader = torch.utils.data.DataLoader(train,
                                                           batch_size=64), torch.utils.data.DataLoader(
        valid, batch_size=64)

    best_loss, path, score = setup_experiment(model, trainloader, validloader, None, args,)
    return best_loss, path, score


def setup_experiment(model, train_loader, valid_loader, resume_checkpoints, args, batcher_test=None):
    '''
   setup the experiment paramaters
   :param model: model class :PreActResnet
   :param train_loader: Batcher
   :param valid_loader: Batcher
   :param resume_checkpoints: str saved checkpoints
   :param args: configs
   :return: best score, and best path string
   '''
    # if resume training:
    if resume_checkpoints:
        print((f'resuming training from {resume_checkpoints}'))
        # redefine the model according to num_outputs              #TODO refactor this model redefinintion to be less redundant
        fc = nn.Linear(model.fc.in_features, args.num_outputs)
        model.fc = fc
        model = load_from_checkpoint(resume_checkpoints, model)

    # setup Trainer params
    params = dict(model=model, lr=args.lr, weight_decay=args.conv_reg, loss_type=args.loss_type,
                  num_outputs=args.num_outputs, metric=args.metric)
    # logging
    wandb.config.update(params)
    # setting experiment_path
    experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                          args.fc_reg, args.conv_reg, args.lr)
    # if doing distribution shift extperiments:
    if args.weight_model:
        class_model = get_model(args.model_name, in_channels=args.in_channels, pretrained=True)
        # redefine the model according to num_outputs
        fc = nn.Linear(class_model.fc.in_features, args.num_outputs)
        class_model.fc = fc
        class_model = load_from_checkpoint(path=args.weight_model, model=class_model)
        experiment = experiment + '_weighted'
    else:
        class_model = None
    # output directory
    dirpath = os.path.join(args.out_dir, experiment)
    print(f'checkpoints directory: {dirpath}')
    os.makedirs(dirpath, exist_ok=True)

    # Trainer
    trainer = Trainer(save_dir=dirpath, **params)

    # Fitting...
    if args.dataset == 'wilds' or args.dataset == 'features':  # attention layer also use this function
        best_loss, path, = trainer.fit_wilds(train_loader, valid_loader, max_epochs=args.max_epochs, gpus=args.gpus,
                                             class_model=class_model)
    else:
        best_loss, path, = trainer.fit(train_loader, valid_loader, max_epochs=args.max_epochs, gpus=args.gpus,
                                       class_model=class_model)
    score = trainer.test(batcher_test)

    return best_loss, path, score


def main(args):
    seed_everything(args.seed)
    # setting experiment_path
    experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                          args.fc_reg, args.conv_reg, args.lr)
    if args.weight_model:
        experiment = experiment + '_weighted'

    dirpath = os.path.join(args.out_dir, experiment)
    os.makedirs(dirpath, exist_ok=True)
    # save data_params for later use
    data_params = dict(dataset=args.dataset, fold=args.fold, ls_bands=args.ls_bands, nl_band=args.nl_band,
                       label_name=args.label_name,
                       nl_label=args.nl_label, include_buildings=args.include_buildings, batch_size=args.batch_size,
                       groupby=args.group)

    params_filepath = os.path.join(dirpath, 'data_params.json')
    with open(params_filepath, 'w') as config_file:
        json.dump(data_params, config_file, indent=4)

    '''
    # saving resnet model params
    params = dict(model_name=args.model_name, in_channels=args.in_channels)
    params_filepath = os.path.join(dirpath, 'params.json')
    with open(params_filepath, 'w') as config_file:
        json.dump(params, config_file, indent=4)
    '''

    wandb.config.update(data_params)
    if args.dataset == 'features':
        MODEL_DIRS = {'resnet_nl_A': 'DHS_OOC_A_encoder_b_nl_geo_b128_fc001_conv001_lre-05'}
        best_loss, best_path, score = geo_attn_exp('resnet_nl', MODEL_DIRS, )
        return
    # dataloader
    elif args.dataset == 'DHS_OOC':
        paths_train = get_paths(args.dataset, 'train', args.fold, args.data_path)

        paths_valid = get_paths(args.dataset, 'val', args.fold, args.data_path)

        paths_test = get_paths(args.dataset, 'test', args.fold, args.data_path)
        print('num_train', len(paths_train))
        print('num_valid', len(paths_valid))
        print('num_valid', len(paths_test))

        paths_train_b = None
        paths_valid_b = None
        paths_test_b = None
        if args.include_buildings:
            paths_train_b = get_paths(args.dataset, 'train', args.fold, args.buildings_records)
            paths_valid_b = get_paths(args.dataset, 'val', args.fold, args.buildings_records)
            paths_test_b = get_paths(args.dataset, 'test', args.fold, args.buildings_records)
            print('b_train', len(paths_train_b))
            print('b_valid', len(paths_valid_b))
            print('b_test', len(paths_test_b))

        # valid=list(paths_train[0:400])+list(paths_train[855:1155])+list(paths_train[1601:2000])
        # train=list(paths_train[400:855])+list(paths_train[1155:1601])+list(paths_train[2000:5000])

        batcher_train = Batcher(paths_train, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                                args.nl_label, args.include_buildings, paths_train_b, args.normalize, args.augment,
                                args.clipn, args.batch_size, groupby=args.group,
                                cache=True, shuffle=True)

        batcher_valid = Batcher(paths_valid, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                                args.nl_label, args.include_buildings, paths_valid_b, args.normalize, False, args.clipn,
                                args.batch_size, groupby=args.group,
                                cache=True, shuffle=False)

        batcher_test = Batcher(paths_test, {'urban_rural': tf.float32}, args.ls_bands, args.nl_band, args.label_name,
                               args.nl_label, args.include_buildings, paths_test_b, args.normalize, False, args.clipn,
                               args.batch_size, groupby='urban',
                               cache=True, shuffle=False)
        '''
        batcher_all= Batcher(get_paths(args.dataset, 'all', args.fold, args.data_path), args.scaler_features_keys,'ms', None, 'wealthpooled',
                                None, False, None, args.normalize, False,
                                False, 8, groupby=args.group,
                                cache=True, shuffle=False)
        
        for i in batcher_all:
            print('large tfrecords: ', i['locs'],i['labels'],i['years'],i['urban_rural'],i['country'])
            break
        '''
    ##############################################################WILDS dataset############################################################
    elif args.dataset == 'wilds':
        dataset = get_dataset(dataset="poverty", download=True, unlabeled=True)

        # Get the training set
        train_data = dataset.get_subset(
            "train",
            # transform=transforms.Compose(
            #     [transforms.Resize((224, 224)), transforms.ToTensor()]
            # ),
        )

        # Prepare the standard data loader
        batcher_train = get_train_loader("standard", train_data, batch_size=64)
        # Get the test set
        test_data = dataset.get_subset(
            "val",
            # transform=transforms.Compose(
            #    [transforms.Resize((224, 224)), transforms.ToTensor()]
            # ),
        )

        # Prepare the data loader
        batcher_valid = get_eval_loader("standard", test_data, batch_size=64)
        batcher_test = None

    ckpt, pretrained = init_model(args.model_init, args.init_ckpt_dir, )
    model_dict = defaultdict()
    encoder_params = defaultdict()

    for (model_key, model_name), in_channels, model_init in zip(args.model_name.items(), args.in_channels,
                                                                args.model_init):
        ckpt, pretrained = init_model(model_init, args.init_ckpt_dir, )
        model_dict[model_key] = get_model(model_name=model_name, in_channels=in_channels, pretrained=pretrained,
                                          ckpt_path=ckpt)

        params = dict(model_name=model_name, in_channels=in_channels)
        encoder_params[model_key] = params
    # saving encoder params
    saved_encoder_params = dict(model_dict=encoder_params, self_attn=args.self_attn)
    encoder_params['self_attn'] = args.self_attn  # comment if not necessary
    encoder_params_filepath = os.path.join(dirpath, 'encoder_params.json')
    print('encoder_params', encoder_params)
    with open(encoder_params_filepath, 'w') as config_file:
        # json.dump(saved_encoder_params, config_file, indent=4)
        json.dump(encoder_params, config_file, indent=4)

    # save the encoder_params

    encoder = Encoder(**model_dict, self_attn=args.self_attn)
    # encoder=Encoder(self_attn=args.self_attn,**model_dict)
    # config = {"lr": args.lr, "wd": args.conv_reg}  # you can remove this now it is for raytune
    best_loss, best_path, score = setup_experiment(encoder, batcher_train, batcher_valid, args.resume, args,
                                                   batcher_test)

    """
    def ray_train(config):
        # instantiating the required variables
        seed_everything(args.seed)
        # setting experiment_path
        experiment = get_full_experiment_name(args.experiment_name, args.batch_size,
                                              args.fc_reg, args.conv_reg, args.lr)
        if args.weight_model:
            experiment = experiment + '_weighted'

        dirpath = os.path.join(args.out_dir, experiment)
        os.makedirs(dirpath, exist_ok=True)
        # save data_params for later use
        data_params = dict(dataset=args.dataset, fold=args.fold, ls_bands=args.ls_bands, nl_band=args.nl_band,
                           label_name=args.label_name,
                           nl_label=args.nl_label, include_buildings=args.include_buildings, batch_size=args.batch_size,
                           groupby=args.group)

        params_filepath = os.path.join(dirpath, 'data_params.json')
        with open(params_filepath, 'w') as config_file:
            json.dump(data_params, config_file, indent=4)

        '''
        # saving resnet model params
        params = dict(model_name=args.model_name, in_channels=args.in_channels)
        params_filepath = os.path.join(dirpath, 'params.json')
        with open(params_filepath, 'w') as config_file:
            json.dump(params, config_file, indent=4)
        '''

        wandb.config.update(data_params)

        # dataloader
        if args.dataset == 'DHS_OOC':
            paths_train = get_paths(args.dataset, 'train', args.fold, args.data_path)

            paths_valid = get_paths(args.dataset, 'val', args.fold, args.data_path)

            paths_test = get_paths(args.dataset, 'test', args.fold, args.data_path)
            print('num_train', len(paths_train))
            print('num_valid', len(paths_valid))
            print('num_valid', len(paths_test))

            paths_train_b = None
            paths_valid_b = None
            paths_test_b = None
            if args.include_buildings:
                paths_train_b = get_paths(args.dataset, 'train', args.fold, args.buildings_records)
                paths_valid_b = get_paths(args.dataset, 'val', args.fold, args.buildings_records)
                paths_test_b = get_paths(args.dataset, 'test', args.fold, args.buildings_records)
                print('b_train', len(paths_train_b))
                print('b_valid', len(paths_valid_b))
                print('b_test', len(paths_test_b))

            # valid=list(paths_train[0:400])+list(paths_train[855:1155])+list(paths_train[1601:2000])
            # train=list(paths_train[400:855])+list(paths_train[1155:1601])+list(paths_train[2000:5000])

            batcher_train = Batcher(paths_train, args.scaler_features_keys, args.ls_bands, args.nl_band,
                                    args.label_name,
                                    args.nl_label, args.include_buildings, paths_train_b, args.normalize, args.augment,
                                    args.clipn, args.batch_size, groupby=args.group,
                                    cache=True, shuffle=True)

            batcher_valid = Batcher(paths_valid, args.scaler_features_keys, args.ls_bands, args.nl_band,
                                    args.label_name,
                                    args.nl_label, args.include_buildings, paths_valid_b, args.normalize, False,
                                    args.clipn, args.batch_size, groupby=args.group,
                                    cache=True, shuffle=False)

            batcher_test = Batcher(paths_test, {'urban_rural': tf.float32}, args.ls_bands, args.nl_band,
                                   args.label_name,
                                   args.nl_label, args.include_buildings, paths_test_b, args.normalize, False,
                                   args.clipn, args.batch_size, groupby='urban',
                                   cache=True, shuffle=False)
        ##############################################################WILDS dataset############################################################
        else:
            dataset = get_dataset(dataset="poverty", download=True)

            # Get the training set
            train_data = dataset.get_subset(
                "train",
                # transform=transforms.Compose(
                #     [transforms.Resize((224, 224)), transforms.ToTensor()]
                # ),
            )

            # Prepare the standard data loader
            train_loader = get_train_loader("standard", train_data, batch_size=64)
            # Get the test set
            test_data = dataset.get_subset(
                "val",
                # transform=transforms.Compose(
                #    [transforms.Resize((224, 224)), transforms.ToTensor()]
                # ),
            )

            # Prepare the data loader
            test_loader = get_eval_loader("standard", test_data, batch_size=64)

        ckpt, pretrained = init_model(args.model_init, args.init_ckpt_dir, )
        model_dict = defaultdict()
        encoder_params = defaultdict()

        for (model_key, model_name), in_channels, model_init in zip(args.model_name.items(), args.in_channels,
                                                                    args.model_init):
            ckpt, pretrained = init_model(model_init, args.init_ckpt_dir, )
            model_dict[model_key] = get_model(model_name=model_name, in_channels=in_channels, pretrained=pretrained,
                                              ckpt_path=ckpt)

            params = dict(model_name=args.model_name, in_channels=args.in_channels)
            encoder_params[model_key] = params

        model_dict.update(dict(self_attn=args.self_attn))
        # saving encoder params
        encoder_params_filepath = os.path.join(dirpath, 'encoder_params.json')
        with open(encoder_params_filepath, 'w') as config_file:
            json.dump(encoder_params.update(dict(self_attn=args.self_attn)), config_file, indent=4)

        # model_dict['self_attn']= MultiHeadedAttention(1,   d_model=512,  dropout=0.1)
        # save the encoder_params

        encoder = Encoder(**model_dict)
        ## END

        best_loss, best_path, score = setup_experiment(encoder, batcher_train, batcher_valid, args.resume, args,config,
                                                       batcher_test)
        #return best_loss, best_path, score
    #config={"lr":args.lr,"wd":args.conv_reg}
    #best_loss, best_path ,score= setup_experiment(encoder,batcher_train, batcher_valid, args.resume,args,config,batcher_test)

   # best_loss, best_path = setup_experiment(model,batcher_train, batcher_test, args.resume, args)
    #best_loss, best_path = setup_experiment(model, train_loader, test_loader, args.resume, args)

    #print(f'Path to best model found during training: \n{best_path}')

    analysis = tune.run(
        ray_train,
        config={

            "lr":  tune.choice([1e-5,1e-4,1e-3]),
            "wd":  tune.choice([1e-4,1e-3,1e-2, 1e-1,1])
        })
    print("Best config: ", analysis.get_best_config(
        metric="mean_loss", mode="min"))
    """


# TODO save hyperparameters .
# TODO Save test scores in csv file

if __name__ == "__main__":
    wandb.init(project=wandp, entity=entity, config={})
    print('GPUS:', torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    args = parse_arguments(parser, default_args)
    args = dotdict(args)

    main(args)
