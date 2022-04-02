# credit goes to https://github.com/sustainlab-group/sustainbench.git https://github.com/sustainlab-group/africa_poverty.git
from __future__ import annotations

import argparse
import os
import warnings
from glob import glob
from typing import Iterable,Optional
import logging
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch


import torchmetrics
from collections import ChainMap

from configs import args

#from batchers.dataset_constants import SURVEY_NAMES,SIZES

from batchers.dataset_constants_buildings import SURVEY_NAMES


class Metric:
    "Metrics dispatcher. Adapted from answer at https://stackoverflow.com/a/58923974"

    def __init__(self, num_classes=None):
        self.num_classes = num_classes

    def get_metric(self, metric='r2'):
        """Dispatch metric with method"""

        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, metric, lambda: "Metric not implemented yet")

        return method()

    def R2(self):
        return torchmetrics.R2Score()

    def r2(self):
        return torchmetrics.PearsonCorrcoef()

    def acc(self):
        return torchmetrics.Accuracy(num_classes=self.num_classes)

    def mse(self):
        return torchmetrics.MeanSquaredError()

    def rank(self):
        return torchmetrics.SpearmanCorrcoef()


def save_results(dir_path: str, np_dict: dict, filename: str
                 ) -> None:
    '''Saves a compressed features.npz file in the given dir.

    Args
    - dir_path: str, path to directory to save .npz file
    - np_dict: dict, maps str => np.array
    - filename: str, name of file to save
    '''
    if not os.path.exists(dir_path):
        print('Creating directory at:', dir_path)
        os.makedirs(dir_path)
    npz_path = os.path.join(dir_path, filename)
    if os.path.exists(npz_path):
        print(f'Path {npz_path} already existed! ,removing old file ....')
        os.remove(npz_path)
    for key, nparr in np_dict.items():
        print(f'{key}: shape {nparr.shape}, dtype {nparr.dtype}')
    print(f'Saving results to {npz_path}')
    np.savez_compressed(npz_path, **np_dict)

def load_npz(path,verbose=True, check=None):
    '''
    loads a npz file into a dictionary
    :param path:
    :param check:dict :dict, key (str) => np.array, values to check
    :return: dict

    '''
    result={}
    with np.load(path) as f :
        for key , value in f.items():
            result[key]=value
            if verbose:
                print(f'key {key} with shape {value.shape}')
        if check:
            for key in check:
                assert key in result
                assert  np.allclose(result[key],check[key])
    return result



def get_full_experiment_name(experiment_name: str, batch_size: int,
                             fc_reg: float, conv_reg: float, lr: float, patch_size):
    if fc_reg < 1:
        fc_str = str(fc_reg).replace('.', '')
        fc_str = fc_str[1:]
    else:
        fc_str = str(fc_reg)
    if conv_reg < 1:
       conv_str = str(conv_reg).replace('.', '')
       conv_str = conv_str[1:]
    else:
        conv_str = str(conv_reg)
    if lr < 1:
        lr_str = str(lr).replace('.', '')
        lr_str = lr_str[1:]
    else:
        lr_str = str(lr)
    return f'{experiment_name}_b{batch_size}_fc{fc_str}_conv{conv_str}_lr{lr_str}_crop{str(patch_size)}'


def check_existing(model_dirs: Iterable[str], outputs_root_dir: str,
                   test_filename: str) -> bool:
    exist = False
    for model_dir in model_dirs:
        dir = os.path.join(outputs_root_dir, model_dir)
        dir = glob(os.path.join(dir, '*ckpt'))
        if len(dir) > 0:
            exist = True
            # check if test file exists
            test_path = os.path.join(model_dir, test_filename)
            if os.path.exists(test_path):
                exist = False
                print(f'found {test_filename} in {model_dir}')

    return exist


def get_paths(dataset: str, split: str, fold: str, root) -> np.ndarray:
    if split == 'all':
        splits = ['train', 'val', 'test']
    else:
        splits=[split]
    paths = []
    fold_name = SURVEY_NAMES[f'{dataset}_{fold}']
    for s in splits:
        for country in fold_name[s]:
            path = os.path.join(root, country + '*', '*.tfrecord.gz')
            paths += glob(path)

    # TODO change to the old data
    #assert  len(paths)==SIZES[f'{dataset}_{fold}'][split]
    return np.sort(paths)


def init_model(method, ckpt_path=None):
    '''

    :param method: str one of ['ckpt', 'imagenet', 'random']
    :param ckpt_path: str checkpoint path
    :return: tuple (ckpt_path:str , pretrained:bool)
    '''
    if method == 'ckpt':
        if ckpt_path:
            return ckpt_path, False
        else:
            raise ValueError('checkpoint path isnot provided')
    elif method == 'imagenet':
        return None, True
    else:
        return None, False

def load_from_checkpoint(path, model):
        print(f'loading the model from saved checkpoint at {path}')
        model.load_state_dict(torch.load(path))
        model.eval()
        return model


log = logging.getLogger(__name__)


def get_sustain_labels(lat, lon, label):
    # strategy 1:write them to tfrecord
    # startegy 2 : read them directly and return them directly
    path=os.path.join(os.getcwd(),'batchers','dhs_final_labels.csv')
    dataframe=pd.read_csv(path)
    match = dataframe[(dataframe['lat'] == lat) & (dataframe['lon'] == lon)]

    if len(match.index)==0:
        print('didnot find label')
        mean = dataframe[label].mean()  # TODO this is so wrong
        return mean
    else:
        print('found label at location')
        return match[label]

def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Helper functions to help with reproducibility of models.
    from pytorch_lightning/utilities/seed.py
    Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
            rank_zero_warn(f"No seed found, seed set to {seed}")
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
                rank_zero_warn(f"Invalid seed found: {repr(env_seed)}, seed set to {seed}")
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        rank_zero_warn(f"{seed} is not in bounds, numpy accepts from {min_seed_value} to {max_seed_value}")
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed

def _warn(*args, stacklevel: int = 2, **kwargs):
    warnings.warn(*args, stacklevel=stacklevel, **kwargs)


def rank_zero_warn(*args, stacklevel: int = 4, **kwargs):
    _warn(*args, stacklevel=stacklevel, **kwargs)


def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)

class dotdict(dict):
    """dot.notation access to dictionary attributes
    Source: How to use a dot “.” to access members of dictionary? \
    https://stackoverflow.com/a/23689767
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_arguments(parser, default_args):
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Run end-to-end training.')

    # paths
    parser.add_argument(
        '--experiment_name', default=default_args.experiment_name,
        help='name of experiment being run')
    parser.add_argument(
        '--out_dir', default=default_args.out_dir,
        help='path to output directory for saving checkpoints and TensorBoard '
             'logs')

    # initialization
    parser.add_argument(
        '--init_ckpt_dir', default=default_args.init_ckpt_dir,
        help='path to checkpoint prefix from which to initialize weights')
    parser.add_argument(
        '--hs_weight_init', choices=[None, 'random', 'same', 'samescaled'],
        default=default_args.hs_weight_init,
        help='method for initializing weights of non-RGB bands in 1st conv '
             'layer')

    # learning parameters
    parser.add_argument(
        '--label_name', default=default_args.label_name,
        help='name of label to use from the TFRecord files')
    parser.add_argument(
        '--batch_size', type=int, default=default_args.batch_size,
        help='batch size')
    parser.add_argument(
        '--augment', action='store_true', default=default_args.augment,
        help='whether to use data augmentation')
    parser.add_argument(
        '--fc_reg', type=float, default=default_args.fc_reg,
        help='Regularization penalty factor for fully connected layers')
    parser.add_argument(
        '--conv_reg', type=float, default=default_args.conv_reg,
        help='Regularization penalty factor for convolution layers')
    parser.add_argument(
        '--lr', type=float, default=default_args.lr,
        help='Learning rate for optimizer')
    parser.add_argument(
        '--lr_decay', type=float, default=default_args.lr_decay,
        help='Decay rate of the learning rate')

    # high-level model control
    parser.add_argument(
        '--model_name', default=default_args.model_name, choices=['resnet18', 'resent34', 'resnet50'],
        help='name of model architecture')

    # data params
    parser.add_argument(
        '--dataset', default=default_args.dataset,
        help='dataset to use')
    parser.add_argument(
        '--fold', default=default_args.fold, choices=['A', 'B', 'C', 'D', 'E'],
        help='fold to use')
    parser.add_argument(
        '--normalize', default=default_args.normalize,
        help='which data to normalize')

    parser.add_argument(
        '--ls_bands', default=default_args.ls_bands,
        #choices=[None, 'rgb', 'ms'],
        help='Landsat bands to use')
    parser.add_argument(
        '--nl_band', default=default_args.nl_band,
        #choices=[None, 'merge', 'split'],
        help='nightlights band')

    # system
    parser.add_argument(
        '--gpu', type=int, default=default_args.gpu,
        help='which GPU to use')
    parser.add_argument(
        '--num_workers', type=int, default=default_args.num_workers,
        help='number of threads for batcher')
    parser.add_argument(
        '--cache', nargs='*', default=default_args.cache, choices=['train', 'train_eval', 'val'],
        help='list of datasets to cache in memory')

    # Misc
    parser.add_argument(
        '--max_epochs', type=int, default=default_args.max_epochs,
        help='maximum number of epochs for training')

    parser.add_argument(
        '--seed', type=int, default=default_args.seed,
        help='seed for random initialization and shuffling')

    parser.add_argument(
        '--lamda', type=int, default=default_args.lamda,
        help='weight of the bias loss')

    args = parser.parse_args()
    args_col = ChainMap(vars(args), vars(default_args))
    return args_col
