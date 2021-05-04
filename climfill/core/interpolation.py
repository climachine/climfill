"""
Copyright 2021 ETH Zurich, contributor: Verena Bessenbacher

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This file gives you the function for filling any missing points
with the spatiotemporal mean of its surrounding points 
(interpolation step) and a convenience function for removing points
outside the area of interest (for example, ocean) from the dataset.
"""

# imports
import numpy as np
import xarray.ufuncs as xu
from scipy.ndimage.filters import generic_filter
from numba_nanmean import nbnanmean 
from scipy import LowLevelCallable
import logging
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)

# fast spatiotemporal mean with cython
mean_fct = LowLevelCallable(nbnanmean.ctypes)

def log_fracmis(data, logtext=''):
    logging.info(f'fraction missing {logtext}: {(np.isnan(data).sum() / data.size).values}')

def gapfill_interpolation(data, n=5):
    """
    Impute (i.e. Gapfill) data by infilling the spatiotemporal mean for each variable independently. A cube of n 5-pixel side length surrounding each missing value is taken and the mean of all non-missing values in this cube is computed and used to infill the missing value . If a point cannot be filled because all the values in the neighbourhood are missing as well, the points is filled by the local monthly climatology. Any remaining missing points are filled by the local temporal mean, or, if not available, the global mean of the variable.

    Parameters
    ----------
    data: xarray dataarray, with coordinates time, latitude, longitude and variable

    n: size of the cube in any spatiotemporal dimension

    Returns
    ----------
    imputed_data: data of the same shape as input data, where all values that were not missing are still the same and all values that were originally missing are imputed via spatiotemporal mean
    """

    # infill spatiotemporal mean
    footprint = np.ones((1,n,n,n))
    tmp = generic_filter(data, mean_fct, footprint=footprint, mode='nearest')
    data = data.fillna(tmp)
    log_fracmis(data, 'after filtering')

    # infill dayofyear mean
    seasonality = dataxr_lost.groupby('time.dayofyear').mean(dim='time')
    data = data.groupby('time.dayofyear').fillna(seasonality).drop('dayofyear')
    log_fracmis(data, 'after seasonality')

    # infill variable mean
    temporal_mean = data.mean(dim=('time'))
    variable_mean = data.mean(dim=('time','latitude','longitude'))
    data = data.fillna(temporal_mean)
    data = data.fillna(variable_mean)
    log_fracmis(data, 'after mean impute')

    return data

def remove_ocean_points(data, landmask):
    """
    Compress data cube by removing all ocean points from the data. The data cube that has the dimensions latitude and longitude will be compressed such that the returned data cube has replaced the dimensions latitude and longitude by the dimension landpoints. Can also be used to remove all points irrelevant for further analysis (region zoom in, remove glaciated regions, only look at one country, only look at ocean,...) 

    Parameters
    ----------
    data: xarray dataarray, coordinates including latitude and longitude 

    landmask: boolean xarray dataarray with only the coordinates latitude and longitude, where grid points on land are True and grid points in the ocean (or regions that are not relevant for research) are False

    Returns
    ----------
    imputed_data: data of the same shape as input data, where all values that were not missing are still the same and all values that were originally missing are imputed via spatiotemporal mean
    """
    landlat, landlon = np.where(landmask)
    return data.isel(longitude=xr.DataArray(landlon, dims='landpoints'), latitude=xr.DataArray(landlat, dims='landpoints'))


if __name__ == '__main__':
    
    data = xr.open_dataset('/path/to/gappy/dataset')
    log_fracmis(data, 'after reading file')

    mask = xu.isnan(data) # create mask of missing values

    data = gapfill_interpolation(data) # initial gapfill all missing values with interpolation
    log_fracmis(data, 'after interpolation') # should be zero

    # optional: remove ocean points for reducing file size
    landmask = xr.open_dataset('/path/to/landmask') # needs dims 'latitude' and 'longitude'
    data = remove_ocean_points(data, landmask)
    mask = remove_ocean_points(mask, landmask)

    # data.to_netcdf ...
    # mask.to_netcdf ...
