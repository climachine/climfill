"""
version on daily resolution, with dask if possible

    @author: verena bessenbacher
    @date: 19 03 2020
"""

# imports
import numpy as np
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

