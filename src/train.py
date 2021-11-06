# Setup Experiment #look at utils/trainer.py
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from batchers.dataset import Batcher
from models.model_generator import get_model
from models.resnet import resnet50
from src.trainer import ResTrain
from utils.utils import get_paths, dotdict, init_model, parse_arguments
from src.configs import args as default_args
from pytorch_lightning import seed_everything


def setup_experiment(args):
    logger = TensorBoardLogger(f"{args.model_name}_logs", name=args.model_name, save_dir=args.out_dir)

    # model
    ckpt, pretrained = init_model(args.model_init, args.init_ckpt_dir, )
    model = get_model(args.model_name,in_channels=len(args.ls_bands+args.nl_band), pretrained=pretrained) ##TEST
    # fc=nn.Linear(model.fc.in_features,1)
    # init_model
    # metric
    # metric = torchmetrics.PearsonCorrcoef()

    # dataloader
    paths_train = get_paths(args.dataset, ['train'], args.fold, '/content/drive/MyDrive/dhs_tfrecords')
    paths_valid = get_paths(args.dataset, ['val'], args.fold, '/content/drive/MyDrive/dhs_tfrecords')

    batcher_train = Batcher(paths_train, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name, args.nl_label, 'DHS', args.augment, args.batch_size,groupby=args.group )
    batcher_valid = Batcher(paths_valid, args.scaler_features_keys, args.ls_bands, args.nl_band, args.label_name, args.nl_label, 'DHS', args.augment, args.batch_size,groupby=args.group )
    # lightning model , trainer

    litmodel = ResTrain(model, args.lr, args.conv_reg, 'regression', 1, metric='r2')
    checkpoint_callback = ModelCheckpoint(monitor=args.monitor, mode=args.mode,
                                          filename="ResnetFineTune-{epoch:02d}-{train-loss:.2f}",
                                          dirpath='/content/drive/MyDrive/checkpoints_resnet',
                                          verbose=True,
                                          save_last=True)

    trainer = pl.Trainer(max_epochs=args.max_epochs, gpus=args.gpu, auto_select_gpus=True,
                         logger=logger,
                         callbacks=[checkpoint_callback],
                         resume_from_checkpoint=ckpt,
                         precision=16,
                         distributed_backend='ddp',
                         profile=True,
                         accumulate_grad_batches=16, )  # understand what it does exactly
    trainer.fit(litmodel, batcher_train, batcher_valid)
    return

def main(args):

    seed_everything(args.seed)
    setup_experiment(args)

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()

        args = parse_arguments(parser, default_args)
        args = dotdict(args)

        main(args)
