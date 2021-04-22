"""
version on daily resolution, with dask if possible

    @author: verena bessenbacher
    @date: 19 03 2020
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

def gapfill_interpolation(data):

    # grace data: special case monthly interpolation
    data.loc['tws'] = (data.loc['tws'].ffill(dim='time').fillna(0) + data.loc['tws'].bfill(dim='time').fillna(0))/2
    log_fracmis(data, 'after grace gapfill')

    # infill spatiotemporal mean
    footprint = np.ones((1,5,5,5))
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

def remove_ocean_points(data, landlon, landlat):
    """
    change data cube from (...,'latitude','longitude') to (...,'landpoints'), efficiently removing all ocean points for reducing the file size.
    """
    return data.isel(longitude=xr.DataArray(landlon, dims='landpoints'), latitude=xr.DataArray(landlat, dims='landpoints'))


if __name__ == '__main__':
    
    data = xr.open_dataset('/path/to/gappy/dataset')
    log_fracmis(data, 'after reading file')

    mask = xu.isnan(data) # create mask of missing values

    data = gapfill_interpolation(data) # initial gapfill all missing values with interpolation
    log_fracmis(data, 'after interpolation') # should be zero

    # optional: remove ocean points for reducing file size
    landmask = xr.open_dataset('/path/to/landmask') # needs dims 'latitude' and 'longitude'
    landlat, landlon = np.where(landmask)
    data = remove_ocean_points(data, landlat, landlon)
    mask = remove_ocean_points(mask, landlat, landlon)

    # data.to_netcdf ...
    # mask.to_netcdf ...
