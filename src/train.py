# Setup Experiment #look at utils/trainer.py
import argparse
import copy
import os.path

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from batchers.dataset import Batcher
from models.model_generator import get_model
from src.trainer import ResTrain
from utils.utils import get_paths, dotdict, init_model, parse_arguments
from src.configs import args as default_args
from pytorch_lightning import seed_everything

#TODO create ouput directory for trained model so that they can be used for extract_features.py

def setup_experiment(model, train_loader, valid_loader, checkpoints, args):
    # logger
    logger = TensorBoardLogger(os.path.join(args.out_dir, f"{args.model_name}_logs"), name=args.model_name)
    # lightning model , trainer

    litmodel = ResTrain(model, args.lr, args.conv_reg, args.loss_type, args.num_outputs, metric='r2')
    checkpoint_callback = ModelCheckpoint(monitor=args.monitor, mode=args.mode,
                                          filename="{args.model_name}-{epoch:02d}-{args.monitor:.2f}",
                                          dirpath='/content/drive/MyDrive/checkpoints_resnet',
                                          verbose=True,
                                          #sanity check
                                          save_last=True)

    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.gpu, auto_select_gpus=True,
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         resume_from_checkpoint=args.resume,
                         precision=16,
                         distributed_backend='ddp',
                         profiler='simple',
                         accumulate_grad_batches=16, )  # understand what it does exactly
    if checkpoints:
        print(f'Initializing using pretrained lightning model:\n{checkpoints}')
        pretrained_model = ResTrain(model, args.lr, args.conv_reg, args.loss_type, args.num_outputs, metric='r2')
        pretrained_model.load_from_checkpoint(checkpoint_path=checkpoints, strict=False)
        litmodel.model = copy.deepcopy(litmodel.model)

    trainer.fit(litmodel, train_loader, valid_loader)

    torch.save(litmodel.state_dict(),
               './models')  # save the model itself (resnetms for example)rather than saving the lighting model

    best_model_ckpt = checkpoint_callback.best_model_path
    best_model_score = checkpoint_callback.best_model_score

    return best_model_ckpt, best_model_score


def main(args):
    seed_everything(args.seed)
    # dataloader
    paths_train = get_paths(args.dataset, ['train'], args.fold, '/content/drive/MyDrive/dhs_tfrecords')
    paths_valid = get_paths(args.dataset, ['val'], args.fold, '/content/drive/MyDrive/dhs_tfrecords')

    batcher_train = Batcher(paths_train, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS', args.augment, args.batch_size, groupby=args.group)
    batcher_valid = Batcher(paths_valid, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name,
                            args.nl_label, 'DHS', args.augment, args.batch_size, groupby=args.group)

    # model
    ckpt, pretrained = init_model(args.model_init, args.init_ckpt_dir, )
    model = get_model(args.model_name, in_channels=args.in_channels, pretrained=pretrained, ckpt_path=ckpt)  ##TEST
    best_model_ckpt, _ = setup_experiment(model, batcher_train, batcher_valid, args.checkpoints, args)
    print(f'Path to best model found during training: \n{best_model_ckpt}')

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        args = parse_arguments(parser, default_args)
        args = dotdict(args)

        main(args)
