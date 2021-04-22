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

lostmask = xu.isnan(dataxr_lost) # missing (lost + orig missing + ocean) is True, rest is False

# extract and save land-sea mask
logging.info('extract land sea mask ...')
lsm = timeinvariant.loc['lsm']
landmask = (lsm.squeeze() > 0.8).load() # land is 1, ocean is 0
landlat, landlon = np.where(landmask)
landmask.to_netcdf(netcdfpath + f'landmask_idebug_{idebugspace}.nc')

# remove ocean points 
logging.info('remove ocean points ...')
dataxr = dataxr.isel(longitude=xr.DataArray(landlon, dims='landpoints'), 
                     latitude=xr.DataArray(landlat, dims='landpoints'))
dataxr_lost = dataxr_lost.isel(longitude=xr.DataArray(landlon, dims='landpoints'), 
                                        latitude=xr.DataArray(landlat, dims='landpoints'))
dataxr_filled = dataxr_filled.isel(longitude=xr.DataArray(landlon, dims='landpoints'), 
                                        latitude=xr.DataArray(landlat, dims='landpoints'))
timeinvariant = timeinvariant.isel(longitude=xr.DataArray(landlon, dims='landpoints'), 
                                        latitude=xr.DataArray(landlat, dims='landpoints'))
lostmask = lostmask.isel(longitude=xr.DataArray(landlon, dims='landpoints'), 
                                        latitude=xr.DataArray(landlat, dims='landpoints'))

dataxr_lost = dataxr_lost.sel(variable=varnames)
dataxr = dataxr.sel(variable=varnames)
dataxr_lost.to_netcdf(netcdfpath + f'dataxr_lost_{missingness}_{frac_missing}_idebug_{idebugspace}.nc') # one day missing in tpmask
dataxr.to_netcdf(netcdfpath + f'dataxr_idebug_{idebugspace}.nc')
log_fracmis(dataxr_filled, 'after remove ocean')
