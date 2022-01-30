from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import ee
import pandas as pd
import time
from tqdm.auto import tqdm


def df_to_fc(df: pd.DataFrame, lat_colname: str = 'lat',
             lon_colname: str = 'lon') -> ee.FeatureCollection:
    '''Create a ee.FeatureCollection from a pd.DataFrame.

    Args
    - csv_path: str, path to CSV file that includes at least two columns for
        latitude and longitude coordinates
    - lat_colname: str, name of latitude column
    - lon_colname: str, name of longitude column

    Returns: ee.FeatureCollection, contains one feature per row in the CSV file
    '''
    # convert values to Python native types
    # see https://stackoverflow.com/a/47424340
    df = df.astype('object')

    ee_features = []
    for i in range(len(df)):
        props = df.iloc[i].to_dict()

        # oddly EE wants (lon, lat) instead of (lat, lon)
        _geometry = ee.Geometry.Point([
            props[lon_colname],
            props[lat_colname],
        ])
        ee_feat = ee.Feature(_geometry, props)
        ee_features.append(ee_feat)

    return ee.FeatureCollection(ee_features)


def add_latlon(img: ee.Image) -> ee.Image:
    '''Creates a new ee.Image with 2 added bands of longitude and latitude
    coordinates named 'LON' and 'LAT', respectively
    '''
    latlon = ee.Image.pixelLonLat().select(
        opt_selectors=['longitude', 'latitude'],
        opt_names=['LON', 'LAT'])
    return img.addBands(latlon)


def tfexporter(collection: ee.FeatureCollection, export: str, prefix: str,
               fname: str, selectors: Optional[ee.List] = None,
               dropselectors: Optional[ee.List] = None,
               bucket: Optional[str] = None) -> ee.batch.Task:
    '''Creates and starts a task to export a ee.FeatureCollection to a TFRecord
    file in Google Drive or Google Cloud Storage (GCS).

    GCS:   gs://bucket/prefix/fname.tfrecord
    Drive: prefix/fname.tfrecord

    Args
    - collection: ee.FeatureCollection
    - export: str, 'drive' for Drive, 'gcs' for GCS
    - prefix: str, folder name in Drive or GCS to export to, no trailing '/'
    - fname: str, filename
    - selectors: None or ee.List of str, names of properties to include in
        output, set to None to include all properties
    - dropselectors: None or ee.List of str, names of properties to exclude
    - bucket: None or str, name of GCS bucket, only used if export=='gcs'

    Returns
    - task: ee.batch.Task
    '''
    if dropselectors is not None:
        if selectors is None:
            selectors = collection.first().propertyNames()

        selectors = selectors.removeAll(dropselectors)

    if export == 'gcs':
        task = ee.batch.Export.table.toCloudStorage(
            collection=collection,
            description=fname,
            bucket=bucket,
            fileNamePrefix=f'{prefix}/{fname}',
            fileFormat='TFRecord',
            selectors=selectors)

    elif export == 'drive':
        task = ee.batch.Export.table.toDrive(
            collection=collection,
            description=fname,
            folder=prefix,
            fileNamePrefix=fname,
            fileFormat='TFRecord',
            selectors=selectors)

    else:
        raise ValueError(f'export "{export}" is not one of ["gcs", "drive"]')

    task.start()
    return task


def sample_patch(point: ee.Feature, patches_array: ee.Image,
                 scale: float) -> ee.Feature:
    '''Extracts an image patch at a specific point.

    Args
    - point: ee.Feature
    - patches_array: ee.Image, Array Image
    - scale: int or float, scale in meters of the projection to sample in

    Returns: ee.Feature, 1 property per band from the input image
    '''
    arrays_samples = patches_array.sample(
        region=point.geometry(),
        scale=scale,
        projection='EPSG:3857',
        factor=None,
        numPixels=None,
        dropNulls=False,
        tileScale=12)
    return arrays_samples.first().copyProperties(point)


def get_array_patches(img: ee.Image,
                      scale: float,
                      ksize: float,
                      points: ee.FeatureCollection,
                      export: str,
                      prefix: str,
                      fname: str,
                      selectors: Optional[ee.List] = None,
                      dropselectors: Optional[ee.List] = None,
                      bucket: Optional[str] = None
                      ) -> ee.batch.Task:
    '''Creates and starts a task to export square image patches in TFRecord
    format to Google Drive or Google Cloud Storage (GCS). The image patches are
    sampled from the given ee.Image at specific coordinates.

    Args
    - img: ee.Image, image covering the entire region of interest
    - scale: int or float, scale in meters of the projection to sample in
    - ksize: int or float, radius of square image patch
    - points: ee.FeatureCollection, coordinates from which to sample patches
    - export: str, 'drive' for Google Drive, 'gcs' for GCS
    - prefix: str, folder name in Drive or GCS to export to, no trailing '/'
    - fname: str, filename for export
    - selectors: None or ee.List, names of properties to include in output,
        set to None to include all properties
    - dropselectors: None or ee.List, names of properties to exclude
    - bucket: None or str, name of GCS bucket, only used if export=='gcs'

    Returns: ee.batch.Task
    '''
    kern = ee.Kernel.square(radius=ksize, units='pixels')
    patches_array = img.neighborhoodToArray(kern)

    # ee.Image.sampleRegions() does not cut it for larger collections,
    # using mapped sample instead
    samples = points.map(lambda pt: sample_patch(pt, patches_array, scale))

    # export to a TFRecord file which can be loaded directly in TensorFlow
    return tfexporter(collection=samples, export=export, prefix=prefix,
                      fname=fname, selectors=selectors,
                      dropselectors=dropselectors, bucket=bucket)


def wait_on_tasks(tasks: Mapping[Any, ee.batch.Task],
                  show_probar: bool = True,
                  poll_interval: int = 20,
                  ) -> None:
    '''Displays a progress bar of task progress.

    Args
    - tasks: dict, maps task ID to a ee.batch.Task
    - show_progbar: bool, whether to display progress bar
    - poll_interval: int, # of seconds between each refresh
    '''
    remaining_tasks = list(tasks.keys())
    done_states = {ee.batch.Task.State.COMPLETED,
                   ee.batch.Task.State.FAILED,
                   ee.batch.Task.State.CANCEL_REQUESTED,
                   ee.batch.Task.State.CANCELLED}

    progbar = tqdm(total=len(remaining_tasks))
    while len(remaining_tasks) > 0:
        new_remaining_tasks = []
        for taskID in remaining_tasks:
            status = tasks[taskID].status()
            state = status['state']

            if state in done_states:
                progbar.update(1)

                if state == ee.batch.Task.State.FAILED:
                    state = (state, status['error_message'])
                elapsed_ms = status['update_timestamp_ms'] - status['creation_timestamp_ms']
                elapsed_min = int((elapsed_ms / 1000) / 60)
                progbar.write(f'Task {taskID} finished in {elapsed_min} min with state: {state}')
            else:
                new_remaining_tasks.append(taskID)
        remaining_tasks = new_remaining_tasks
        time.sleep(poll_interval)
    progbar.close()

"""
class AfricaBuildings:
     def __init__(self,prop) -> None:
        '''
        Args
        - filterpoly: ee.Geometry
        - start_date: str, string representation of start date
        - end_date: str, string representation of end date
        '''
        self.t = ee.FeatureCollection('GOOGLE/Research/open-buildings/v1/polygons')
        self.t_gte_070 = self.t.filter('confidence >= 0.70')
        #self.t_gte_300 = self.t.filter('area_in_meters >300')
        t_img = self.t_gte_070.reduceToImage([prop], ee.Reducer.first()).unmask(0)
        t_img_rescaled = t_img.setDefaultProjection('EPSG:3857').reduceResolution(reducer=ee.Reducer.mean(), maxPixels=900)
        self.t_img = t_img_rescaled.rename(['buildings'])
     def add_layer(self,prop)-> None:
         c_image=self.t.reduceToImage([prop],ee.Reducer.first()).unmask(0)
         c_img_rescaled = c_image.setDefaultProjection('EPSG:3857').reduceResolution(reducer=ee.Reducer.mean(),
                                                                                   maxPixels=900)
         c_img_rescaled=c_img_rescaled.rename(['buildings_conf'])
         self.t_img.addBands(c_img_rescaled)
"""

class AfricaBuildings:
    def __init__(self, prop) -> None:
        self.t = ee.FeatureCollection('GOOGLE/Research/open-buildings/v1/polygons')
        self.t_gte_070 = self.t.filter('confidence >= 0.70')
        # self.t_gte_300 = self.t.filter('area_in_meters >300')
        t_img = self.t_gte_070.reduceToImage([prop], ee.Reducer.first()).unmask(0)
        t_img_rescaled = t_img.setDefaultProjection('EPSG:3857').reduceResolution(reducer=ee.Reducer.mean(), maxPixels=900)
        self.t_img = t_img_rescaled.rename(['buildings'])
