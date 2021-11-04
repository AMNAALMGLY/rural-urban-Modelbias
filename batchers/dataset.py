# credit goes to https://github.com/sustainlab-group/sustainbench.git https://github.com/sustainlab-group/africa_poverty.git
from __future__ import annotations
import tensorflow_datasets as tfds

from pathlib import Path
import tensorflow as tf
from collections.abc import Iterable, Mapping
from typing import Optional
import pickle
import os
from glob import glob
import numpy as np
import pandas as pd
import torch
from dataset_constants import MEANS_DICT, STD_DEVS_DICT
from utils.utils import  save_results
from collections import defaultdict

AUTO = tf.data.experimental.AUTOTUNE


class Batcher():
    """
    The PovertyMap poverty measure prediction dataset Iterator.
    This is a processed version of LandSat 5/7/8 Surface Reflectance,
    DMSP-OLS, and VIIRS Nightlights satellite imagery originally
    from Google Earth Engine under the names
        Landsat 8: `LANDSAT/LC08/C01/T1_SR`
        Landsat 7: `LANDSAT/LE07/C01/T1_SR`
        Landsat 5: `LANDSAT/LT05/C01/T1_SR`
         DMSP-OLS: `NOAA/DMSP-OLS/CALIBRATED_LIGHTS_V4`
            VIIRS: `NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG`.
    The labels come from surveys conducted through the DHS Program:
        https://dhsprogram.com/data/available-datasets.cfm
    Input (x):
        224 x 224 x 8 satellite image, with 7 channels from Landsat and
        1 nighttime light channel from DMSP/VIIRS. These images have not been
        mean / std normalized.
    Output (y):
        y is a real-valued asset wealth index. Higher value corresponds to more
        asset wealth.
    Metadata:
        Each image is annotated with location coordinates (lat/lon, noised for
        anonymity), survey year, urban/rural classification, country.
    License:
        LandSat/DMSP/VIIRS data is U.S. Public Domain.
    TODO:citation
    """

    def __init__(self, tfrecords, scalar_features_keys, ls_bands, nl_bands, label, nl_label, normalize, augment,
                 batch_size, save_dir=None):

        '''
        initializes the loader as follows :

        '''
        self.tfrecords = tfrecords
        self.scalar_features_keys = scalar_features_keys
        self.ls_bands = ls_bands
        self.nl_bands = nl_bands
        self.label = label
        self.nl_label = nl_label
        self.normalize = normalize
        self.augment = augment
        self.save_dir = save_dir

        self.batch_size = batch_size
        self._iterator = None

        # TODO:chech values of arguments passed

    '''Not necessary anymore
    def __getitem__(self, idx):
      ##TODO
        path = self.save_dir + f'{idx}'
        with np.load(path) as data:
          return data
    '''

    def __len__(self):
        '''
         length of the iterator
        '''
        nbatches = len(self.tfrecords) // self.batch_size
        if len(self.tfrecords) % self.batch_size != 0:
            nbatches += 1
        return nbatches

    def groupby(self):
        '''
        group by urban/rural

        '''
        raise NotImplementedError

    def tfrecords_to_dict(self, example: tf.Tensor) -> dict[str, tf.Tensor]:
        '''
        convert the records to dictionaries of {'feature_name' : tf.Tensor}

        '''
        # TODO:NIGHTLIGHT band/label

        bands = {'rgb': ['BLUE', 'GREEN', 'RED'], 'ms': ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR'],
                 'merge': ['NIGHTLIGHTS'], 'split': ['NIGHTLIGHTS'], 'center': ['NIGHTLIGHTS'], 'mean': ['NIGHTLIGHTS']}
        keys_to_features = {}
        scalar_float_keys = ['lat', 'lon', 'year']
        if self.label is not None:
            scalar_float_keys.append(self.label)
        ex_bands = bands[self.ls_bands] + bands.get(self.nl_bands, [])
        print('ex_bands :', ex_bands)
        for band in ex_bands:
            keys_to_features[band] = tf.io.FixedLenFeature(shape=[255 ** 2], dtype=tf.float32)

        for key in scalar_float_keys:
            keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=tf.float32)
        if self.scalar_features_keys is not None:
            for key, dtype in self.scalar_features_keys.items():
                keys_to_features[key] = tf.io.FixedLenFeature(shape=[], dtype=dtype)

        ex = tf.io.parse_single_example(example, features=keys_to_features)

        loc = tf.stack([ex['lat'], ex['lon']])
        year = tf.cast(ex.get('year', -1), tf.int32)
        img = float('nan')
        if len(ex_bands) > 0:
            for band in ex_bands:  ##TODO is this loop necessary ?vectorize
                ex[band].set_shape([255 * 255])
                ex[band] = tf.reshape(ex[band], [255, 255])[15:-16, 15:-16]  # crop to 224x224

                if self.normalize:
                    means = MEANS_DICT[self.normalize]
                    stds = STD_DEVS_DICT[self.normalize]
                    if band == 'NIGHTLIGHTS' and year < 2012:
                        ex[band] = (ex[band] - means['DMSP']) / std_devs['DMSP']
                    elif band == 'NIGHTLIGHTS' and year > 2012:
                        ex[band] = (ex[band] - means['VIIRS']) / std_devs['VIIRS']
                    else:
                        ex[band] = ex[band] - means[band] / stds[band]
            # TODO augmentation
            img = tf.stack([ex[band] for band in ex_bands], axis=2)

        else:
            raise ValueError

        if self.label:
            label_ms = ex.get(self.label, float('nan'))

        if self.nl_label:
            if self.nl_label == 'mean':
                nl = tf.reduce_mean(ex['NIGHTLIGHTS'])
            elif self.nl_label == 'center':
                nl = ex['NIGHTLIGHTS'][112, 112]
            else:
                raise ValueError  ##TODO : move this to  the initialization (restrict to range of values)
        else:
            nl = float('nan')

        if self.nl_label and self.label:
            labels = tf.stack([label_ms, nl])
        elif self.nl_label:
            labels = nl
        elif self.label:
            labels = label_ms
        else:
            labels = float('nan')

        result = {'images': img, 'locs': loc, 'years': year, 'labels': labels}

        if self.scalar_features_keys:
            for key in self.scalar_features_keys:
                result[key] = ex[key]
        return result

    def tfDatase_to_np(self):
        # TODO:test, move to utils.py
        '''
        convert te dataset to .npz if needed and saved in self.save_dir
        '''
        ds = tf.data.TFRecordDataset(self.tfrecords, num_parallel_reads=AUTO, compression_type='GZIP')

        ds = ds.map(lambda ex: self.tfrecords_to_dict(ex))
        idx = 0
        for record in ds.as_numpy_iterator():
            save_results(self.save_dir, record, f'{idx:05d}')
            idx += 1

    def get_dataset(self, cache=None, shuffle=False):
        '''
        do the tf_to dict operation to the whole dataset in numpy dtype
        '''

        dataset = tf.data.TFRecordDataset(self.tfrecords, num_parallel_reads=AUTO, compression_type='GZIP')
        if cache:
            dataset = dataset.cache()
        if shuffle:
            # shuffle the order of the input files, then interleave their individual records
            dataset = tf.data.Dataset.from_tensor_slices(self.tfrecord_files)
            dataset = dataset.shuffle(buffer_size=1000)
            dataset = dataset.apply(
                tf.data.experimental.parallel_interleave(
                    lambda filename: tf.data.TFRecordDataset(filename),
                    cycle_length=4))  ##TODO edit this cycle lenght

        dataset = dataset.map(lambda ex: self.tfrecords_to_dict(ex))
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(AUTO)
        return tfds.as_numpy(dataset)

    def __iter__(self):
        '''
        implement iterator of the  loader
        '''
        self.ds = self.get_dataset()
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch
