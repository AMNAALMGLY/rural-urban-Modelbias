# credit goes to https://github.com/sustainlab-group/sustainbench.git https://github.com/sustainlab-group/africa_poverty.git
# https://www.kaggle.com/hidehisaarai1213/g2net-read-from-tfrecord-train-with-pytorch
from __future__ import annotations

from typing import Callable, Tuple

import tensorflow as tf
import os
from batchers.dataset_constants import MEANS_DICT, STD_DEVS_DICT
from batchers.dataset_constants_buildings import MAX_DICT, MIN_DICT
from configs import args

from utils.utils import save_results
import time

AUTO: int = tf.data.experimental.AUTOTUNE

# choose which GPU to run on
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seed = 123


# TODO split nl_band function
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
    This class is an Iterable dataset throught which you can iterate directly through data
    Example :
    b=Batcher(**args)
    for batch in b:
      image=b['images']
      label=b['labels']

    """

    def __init__(self, tfrecords, scalar_features_keys, ls_bands, nl_bands, label, nl_label,include_buildings=False, buildings_records=None,
                 normalize='DHS',
                 augment=False, clipn=True,
                 batch_size=64, groupby=None, cache=None, shuffle=False, save_dir=None):

        '''
        initializes the loader as follows :
         Args
        - tfrecords: list of str,
            - path(s) to TFRecord files containing satellite images
        - label_name: str, name of feature within TFRecords to use as label, or None
        - scalar_features: dict, maps names (str) of additional features within a TFRecord
            to their parsed types
        - ls_bands: one of [None, 'rgb', 'ms'], which Landsat bands to include in batch['images']
            - None: no Landsat bands
            - 'rgb': only the RGB bands
            - 'ms': all 7 Landsat bands
        - nl_band: one of [None, 'merge', 'split'], which NL bands to include in batch['images']
            - None: no nightlights band
            - 'merge': single nightlights band
            - 'split': separate bands for DMSP and VIIRS (if one is absent, then band is all 0)
        - nl_label: one of [None, 'center', 'mean']
            - None: do not include nightlights as a label
            - 'center': nightlight value of center pixel
            - 'mean': mean nightlights value
        - include_buildings:bool whether or not to include building_layer
        - normalize: str, must be one of the keys of MEANS_DICT
           - if given, subtracts mean and divides by std-dev
        - augment: bool, whether to use data augmentation, should be False when not training
        - clipneg: bool, whether to clip negative values to 0
        - batch_size: int
        -groupby: str one of ['urban' , 'rural'] to get iterator over one of them

        - cache: bool, whether to cache this dataset in memory
        - shuffle: bool, whether to shuffle data, should be False when not training
        - save_dir: str, if data is converted to np

        '''
        self.tfrecords = tfrecords
        self.scalar_features_keys = scalar_features_keys
        self.ls_bands = ls_bands
        self.nl_bands = nl_bands
        self.label = label
        self.nl_label = nl_label
        self.include_buildings=include_buildings
        self.buildings_records=buildings_records
        self.normalize = normalize
        self.augment = augment
        self.clipn = clipn
        self.groupby = groupby
        self.cache = cache
        self.save_dir = save_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._iterator = None
        self.ds = self.get_dataset()



        # TODO:check values of arguments passed

    def __len__(self):
        '''
         length of the iterator(number of batches given the length of tfrecords and batch_size)
         :return:int length
        '''
        if self.groupby is None:
            nbatches = len(self.tfrecords) // self.batch_size
            if len(self.tfrecords) % self.batch_size != 0:
                nbatches += 1
            return nbatches
        else:
            # TODO: save the length in a static variable (less time)
            return sum(1 for i in self)

    def b_tfrecords_to_dict(self, example: tf.Tensor) -> dict[str, tf.Tensor]:
        '''
        custom function that converts building tfrecord into dict
        :param example: tf dataset example
        :return: dict
        '''
        band='buildings'
        keys_to_features = {}
        keys_to_features[band] = tf.io.FixedLenFeature(shape=[355 ** 2], dtype=tf.float32)
        ex = tf.io.parse_single_example(example, features=keys_to_features)
        ex[band].set_shape([355 * 355])
        ex[band] = tf.reshape(ex[band], [355, 355,1])
        #ex[band]=tf.expand_dims(ex[band],axis=-1)
        #print('size before reshape ',ex[band].shape)
        #ex[band] = tf.image.resize(ex[band], [224, 224 ])
        #ex[band] = tf.reshape(ex[band], [255, 255])[15:-16, 15:-16]  # crop to 224x224

        if self.clipn:
            ex[band] = tf.nn.relu(ex[band])
        #return {'buildings':tf.expand_dims(ex[band],axis=-1)}
        return {'buildings':ex[band]}

    def tfrecords_to_dict(self, example: tf.Tensor) -> dict[str, tf.Tensor]:

        '''
        convert the records to dictionaries of {'feature_name' : tf.Tensor}
        Args
        - example: a tf.train.Example

        Returns: dict {'images': img, 'labels': label, 'locs': loc, 'years': year, ...}
        - img: tf.Tensor, shape [224, 224, C], type float32
          - channel order is [B, G, R, SWIR1, SWIR2, TEMP1, NIR, NIGHTLIGHTS]
        - label: tf.Tensor, scalar or shape [2], type float32
          - not returned if both self.label_name and self.nl_label are None
          - [label, nl_label] (shape [2]) if self.label_name and self.nl_label are both not None
          - otherwise, is a scalar tf.Tensor containing the single label
        - loc: tf.Tensor, shape [2], type float32, order is [lat, lon]
        - year: tf.Tensor, scalar, type int32
          - default value of -1 if 'year' is not a key in the protobuf
        - may include other keys if self.scalar_features is not None
        '''

        # TODO:NIGHTLIGHT band/label

        bands = {'rgb': ['BLUE', 'GREEN', 'RED'], 'ms': ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR'],
                 'merge': ['NIGHTLIGHTS'], 'split': ['NIGHTLIGHTS'], 'center': ['NIGHTLIGHTS'], 'mean': ['NIGHTLIGHTS'],'geo':['LON','LAT']}

        keys_to_features = {}
        scalar_float_keys = ['lat', 'lon', 'year']

        if self.label is not None:
            scalar_float_keys.append(self.label)

        ex_bands= bands.get(self.ls_bands, []) + bands.get(self.nl_bands, [])


        print('ex_bands :', ex_bands)
        for band in ex_bands:
            keys_to_features[band] = tf.io.FixedLenFeature(shape=[355 ** 2], dtype=tf.float32)

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
                ex[band].set_shape([355 * 355])
                #ex[band].set_shape([448*448])
                #ex[band] = tf.image.resize_with_crop_or_pad(ex[band], 3, 3)
                #ex[band] = tf.reshape(ex[band], [448, 448])
                #ex[band] = tf.reshape(ex[band], [355, 355])[65:-66, 65:-66]  # crop to 224x224
                ex[band] = tf.reshape(ex[band], [355, 355 ,1])
                #[15:-16, 15:-16]
                #ex[band]=tf.image.resize(ex[band],[224,224])
                #[65:-66, 65:-66]
                #RESIZE FOR merging with building
                #ex[band]=tf.image.resize(ex[band], [355, 355], method='nearest')
                if self.clipn:
                    ex[band] = tf.nn.relu(ex[band])
                if self.normalize:
                    means = MEANS_DICT[self.normalize]
                    stds = STD_DEVS_DICT[self.normalize]
                    mins=MIN_DICT[self.normalize]
                    maxs=MAX_DICT[self.normalize]

                    if band == 'NIGHTLIGHTS':
                        #all_0 = tf.zeros(shape=[224, 224], dtype=tf.float32, name='all_0')
                        ex[band] = tf.cond(
                            year < 2012,  # true = DMSP
                            #true_fn=lambda:  ex[band] = (ex[band] - mins['DMSP']) / (maxs['DMSP']-mins['DMSP'])
                            #false_fn=lambda:  ex[band] = (ex[band] - mins['VIIRS']) / (maxs['VIIRS']-mins['VIIRS'])
                            true_fn=lambda: (ex[band] - mins['DMSP']) / maxs['DMSP'],
                            #true_fn=lambda: all_0,
                            false_fn=lambda: (ex[band] - mins['VIIRS']) / maxs['VIIRS'])

                    else:
                        ex[band] = (ex[band] - mins[band]) / (maxs[band]-mins[band])
                        #ex[band] = (ex[band] - means[band]) / stds[band]

            img = tf.concat([ex[band] for band in ex_bands], axis=2)


        else:
            img=img

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
                #if key=='country':
                 #   result[key]=result[key].numpy().decode('utf-8')
        print('finished converting to dict')

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

    # do the tf_to dict operation to the whole dataset in numpy dtype

    def get_dataset(self,):
        '''
        performs the all steps of mapping the tfrecords to dictionaries then preprocess , and output the dataset divided by batches

        :return: tensorflow dataset divided into batches
        '''
        start = time.time()

        if self.shuffle  and args.buildings_records is  None: #shuffle only if you don't want to include building dataset
            print('in shuffle')
            # shuffle the order of the input files, then interleave their individual records
            dataset = tf.data.Dataset.from_tensor_slices(self.tfrecords)\
                .shuffle(buffer_size=1000,reshuffle_each_iteration=True).\
                interleave(
                 lambda file_path: tf.data.TFRecordDataset(file_path,  compression_type='GZIP',num_parallel_reads=AUTO),
                cycle_length=5,
                block_length=1,
                num_parallel_calls=AUTO
            )


        else:
            # convert to individual records
            dataset = tf.data.TFRecordDataset(
                filenames=self.tfrecords,
                compression_type='GZIP',
                buffer_size=1024 * 1024 * 128,  # 128 MB buffer size
                num_parallel_reads=AUTO)




        dataset = dataset.prefetch(2 * self.batch_size)

        dataset = dataset.map(lambda ex: self.tfrecords_to_dict(ex), num_parallel_calls=AUTO)
        if self.nl_bands == 'split':
            dataset = dataset.map(self.split_nl_band)


        if self.groupby == 'urban':

            dataset = dataset.filter(lambda ex: tf.equal(ex['urban_rural'], 1.0))
        elif self.groupby == 'rural':
            dataset = dataset.filter(lambda ex: tf.equal(ex['urban_rural'], 0.0))

        if self.include_buildings :
            #even if there is no ls or nl bands , we want labels from the other dataset
            b_dataset = tf.data.TFRecordDataset(
                filenames=self.buildings_records,
                compression_type='GZIP',
                buffer_size=1024 * 1024 * 128,  # 128 MB buffer size
                num_parallel_reads=AUTO)
            b_dataset = b_dataset.prefetch(2 * self.batch_size)
            b_dataset = b_dataset.map(lambda ex: self.b_tfrecords_to_dict(ex), num_parallel_calls=AUTO)

            dataset = tf.data.Dataset.zip((dataset, b_dataset))



        if self.cache:
            dataset = dataset.cache()
            print('in cahce')

        if self.shuffle:
            buffer_size=1000
            dataset = dataset.shuffle(buffer_size=buffer_size,reshuffle_each_iteration=True)


        if self.augment:
            print('in augment')
            counter = tf.data.experimental.Counter()
            dataset = tf.data.Dataset.zip((dataset, (counter, counter)))
            if self.include_buildings  :
                if self.nl_bands or self.ls_bands:         #TODO modify for brightness for ms augmentation
                    dataset = dataset.map(self.b_augment, num_parallel_calls=AUTO)
                else:
                    dataset=dataset.map(self.build_augment,num_parallel_calls=AUTO)

            else:
                dataset = dataset.map(self.augment_ex, num_parallel_calls=AUTO)

        dataset = dataset.batch(batch_size=self.batch_size)
        print('in batching')


        dataset = dataset.prefetch(2)
        print(f'Time in getdataset: {time.time() - start}')
        '''
        iterator = iter(dataset)
        batch = next(iterator)
        # iter_init = iterator.initializer
        return batch
        '''
        return dataset


    def split_nl_band(self, ex: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
        '''Splits the NL band into separate DMSP and VIIRS bands.

        Args
        - ex: dict {'images': img, 'years': year, ...}
            - img: tf.Tensor, shape [H, W, C], type float32, final band is NL
            - year: tf.Tensor, scalar, type int32

        Returns: ex, with img updated to have 2 NL bands
        - img: tf.Tensor, shape [H, W, C], type float32, last two bands are [DMSP, VIIRS]
        '''
        assert self.nl_bands == 'split'
        all_0 = tf.zeros(shape=[224, 224, 1], dtype=tf.float32, name='all_0')
        img = ex['images']
        year = ex['years']

        ex['images'] = tf.cond(
            year < 2012,
            # if DMSP, then add an all-0 VIIRS band to the end
            #true_fn=lambda: tf.concat([ img,all_0,], axis=2),
            true_fn=lambda: tf.concat([img[:, :, 0:-1], all_0, img[:, :, -1:]], axis=2),
            # if VIIRS, then insert an all-0 DMSP band before the last band
            false_fn=lambda: tf.concat([img[:, :, 0:-1], all_0, img[:, :, -1:]], axis=2)
        )
        return ex



    def augment_ex(self, ex: dict[str, tf.Tensor], seed) -> dict[str, tf.Tensor]:
 
        print('in augment ex')
        img = ex['images']
        img = tf.image.stateless_random_flip_left_right(img, seed=seed)
        img = tf.image.stateless_random_flip_left_right(img, seed=seed)

        #img=tf.image.stateless_random_crop(img, size=[210, 210, args.in_channels], seed=seed)
        #img=tf.image.central_crop(img,0.8)
        #img=tf.image.rot90(img)
        #img=tf.image.resize_with_crop_or_pad(img, 448, 448)


        if self.nl_bands and self.ls_bands:
            if self.nl_bands == 'merge':

                img = tf.image.stateless_random_brightness(img[:, :, :-1], max_delta=0.5, seed=seed)
                img = tf.image.stateless_random_contrast(img, lower=0.75, upper=1.25, seed=seed)

                img = tf.concat([img, ex['images'][:, :, -1:]], axis=-1)
                print(img.shape)
            else:

                img = tf.image.stateless_random_brightness(img[:, :, :-2], max_delta=0.5, seed=seed)
                img = tf.image.stateless_random_contrast(img, lower=0.75, upper=1.25, seed=seed)
                #img = tf.image.stateless_random_saturation(img[:, :, :-2], lower=0.75, upper=1.25, seed=seed)
                #img = tf.image.stateless_random_hue(img[:, :, :-2], max_delta=0.1, seed=seed)
                img = tf.concat([img, ex['images'][:, :, -2:]], axis=-1)

        elif self.ls_bands:

            img = tf.image.stateless_random_brightness(img, max_delta=0.5, seed=seed)
            img = tf.image.stateless_random_contrast(img, lower=0.75, upper=1.25, seed=seed)
          #  img=tf.image.stateless_random_saturation(img, lower=0.75, upper=1.25,seed=seed)
           # img=tf.image.stateless_random_hue(img,max_delta=0.1,seed=seed)

           #img= tf.image.random_brightness(img, max_delta=0.5)
           #img = tf.image.random_contrast(img, lower=0.75, upper=1.25)
        print('images augment')

        ex['images'] = img
        return ex
    def b_augment(self,ex,seed):
              #TODO validate this
        img=ex[0]['images']
        b=ex[1]['buildings']
        #print(img.shape,b.shape)
        img = tf.image.stateless_random_flip_left_right(img, seed=seed)
        img = tf.image.stateless_random_flip_left_right(img, seed=seed)
       # img=tf.image.stateless_random_crop(
        #    img, size=[210, 210, args.in_channels-1], seed=seed)
        b = tf.image.stateless_random_flip_left_right(b, seed=seed)
        b = tf.image.stateless_random_flip_left_right(b, seed=seed)
      #  b = tf.image.stateless_random_crop(
      #      b, size=[210, 210, 1], seed=seed)
        print('images augment')

        ex[0]['images'] =img
        ex[1]['buildings']=b
        #print('afterAug',ex[1]['buildings'])
        return ex
    def build_augment(self,ex,seed):
        b=ex[1]['buildings']
        b = tf.image.stateless_random_flip_left_right(b, seed=seed)
        b = tf.image.stateless_random_flip_left_right(b, seed=seed)
        ex[1]['buildings']=b
        return ex


    '''
    def augment_ex(self, ex: dict[str, tf.Tensor]) -> dict[str, tf.Tensor]:
   
        assert self.augment
        img = ex['images']

        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_flip_left_right(img)
        img = self.augment_levels(img)

        ex['images'] = img
        return ex

    def augment_levels(self, img: tf.Tensor) -> tf.Tensor:
   
        def rand_levels(image: tf.Tensor) -> tf.Tensor:
            # up to 0.5 std dev brightness change
            image = tf.image.random_brightness(image, max_delta=0.5)
            image = tf.image.random_contrast(image, lower=0.75, upper=1.25)
            return image

        # only do random brightness / contrast on non-NL bands
        if self.ls_bands is not None:
            if self.nl_bands is None:
                img = rand_levels(img)
            elif self.nl_bands == 'merge':
                img_nonl = rand_levels(img[:, :, :-1])
                img = tf.concat([img_nonl, img[:, :, -1:]], axis=2)
            elif self.nl_bands == 'split':
                img_nonl = rand_levels(img[:, :, :-2])
                img = tf.concat([img_nonl, img[:, :, -2:]], axis=2)
        return img
    '''
    def __iter__(self):
        '''
        implement iterator of the  dataset
        :return:dataset iterator as a numpy iterator
        '''
        start = time.time()
        # self.ds=self.get_dataset()
        if self._iterator is None:
            self._iterator = iter(self.ds.as_numpy_iterator())
        else:
            self._reset()

        print(f'time in iter: {time.time() - start}')
        return self._iterator

    def _reset(self):
        '''
        resets the iterator every new iteration
        :return: numpy iterator
        '''
        self._iterator = iter(self.ds.as_numpy_iterator())


    def __next__(self):
        '''
        goes to the next batch
        :return: batch as a dictionary [str:ndarray]
        '''
        start = time.time()
        batch = next(self._iterator)
        print(f'time in next: {time.time() - start}')
        return batch

