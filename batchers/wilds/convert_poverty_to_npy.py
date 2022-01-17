'''
Adapted from github.com/sustainlab-group/africa_poverty/data_analysis/dhs.ipynb
'''
import tensorflow as tf
import numpy as np
from batchers.wilds import batcher
from batchers.wilds import dataset_constants_buildings
from tqdm import tqdm

FOLDS = ['A', 'B', 'C', 'D', 'E']
SPLITS = ['train', 'val', 'test']
BAND_ORDER = ['BLUE', 'GREEN', 'RED', 'SWIR1', 'SWIR2', 'TEMP1', 'NIR', 'NIGHTLIGHTS']
DATASET = '2009-17'

COUNTRIES = np.asarray(dataset_constants_buildings.DHS_COUNTRIES)


def get_images(tfrecord_paths, label_name='wealthpooled', return_meta=False):
    '''
    Args
    - tfrecord_paths: list of str, length N <= 32, paths of TFRecord files

    Returns: np.array, shape [N, 224, 224, 8], type float32
    '''
    batch = batcher.Batcher(
        tfrecord_files=tfrecord_paths,
        dataset=DATASET,
        batch_size=32,
        ls_bands='buildings',
        nl_band=None,
        label_name=label_name,
        shuffle=False,
        augment=False,
        negatives='zero',
        normalize=True).get_batch()
    '''
    #with tf.compat.v1.Session() as sess:
    for elem in dataset:
        sess.run(init_iter)
        if return_meta:
            ret = sess.run(batch_op)
        else:
            ret = sess.run(batch_op['images'])
    return ret
    '''
    return batch['images']

if __name__ == '__main__':
    tfrecord_paths = np.asarray(batcher.get_tfrecord_paths(dataset=DATASET, split='all',fold='A'))

    num_batches = len(tfrecord_paths) // 32
    if len(tfrecord_paths) % 32 != 0:
        num_batches += 1

    imgs = []

    for i in tqdm(range(num_batches)):
        imgs.append(get_images(tfrecord_paths[i*32: (i+1)*32]))

    imgs = np.concatenate(imgs, axis=0)
    np.save('./landsat_poverty_imgs.npy', imgs)


