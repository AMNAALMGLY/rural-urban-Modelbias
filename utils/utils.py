
# credit goes to https://github.com/sustainlab-group/sustainbench.git https://github.com/sustainlab-group/africa_poverty.git
from __future__ import annotations

import os
from glob import glob

import numpy as np
import tensorflow as tf

from batchers.dataset_constants import SURVEY_NAMES
import torchmetrics

AUTO = tf.data.experimental.AUTOTUNE



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
    assert not os.path.exists(npz_path), f'Path {npz_path} already existed!'
    for key, nparr in np_dict.items():
        print(f'{key}: shape {nparr.shape}, dtype {nparr.dtype}')
    print(f'Saving results to {npz_path}')
    np.savez_compressed(npz_path, **np_dict)


def get_paths(dataset: str, split: str, fold: str, root) -> np.ndarray:
    if split == 'all':
        split = ['train', 'val', 'test']
    paths = []
    fold = SURVEY_NAMES[f'{dataset}_{fold}']
    for s in split:
        for country in fold[s]:
            path = os.path.join(root, country + '*', '*.tfrecord.gz')
            paths += glob(path)
    return np.sort(paths)

