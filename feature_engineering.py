"""
version on daily resolution, with dask if possible

    @author: verena bessenbacher
    @date: 19 03 2020
"""

# imports
import sys
import numpy as np
import xarray as xr
import xarray.ufuncs as xu
import logging
import argparse
logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG)
logging.getLogger('regionmask').setLevel(logging.WARNING)
logging.getLogger('fiona').setLevel(logging.WARNING)
import regionmask
from namelist import plotpath, years, largefilepath, varnames, invarnames
from scipy.ndimage.filters import generic_filter

def log_fracmis(data, logtext=''):
    logging.info(f'fraction missing {logtext}: {(np.isnan(data).sum() / data.size).values}')

def logscale_precip(data):
    data.loc['tp'] = data.loc['tp'].where(data.loc['tp'] > 0.000001, -1)
    data.loc['tp'] = xu.log(data.loc['tp'])
    data.loc['tp'] = data.loc['tp'].where(~np.isnan(data.loc['tp']), -20) # all zero precip got -1, all -1 got nan, all nan get -20
    return data

def normalise(data, isave=False):
    """ mean zero unit (1) standard deviation """
    datamean = data.mean(dim=('time','landpoints'))
    datastd = data.std(dim=('time','landpoints'))
    data = (data - datamean) / datastd
    #datamean = datamean.sel(variable=varnames)
    #datastd = datastd.sel(variable=varnames)
    return data, datamean, datastd

def stack(data):
    # add select variable to remove it in gapfill?
    return data.stack(datapoints=('time','landpoints')).reset_index('datapoints')

# steps of the feature engineering

# log-scale precip

# add latitude, longitude, time

# add lags

# normalise

# stack constant maps and variables

# stack to datapoints

# divide precip into bool "itrained" and logscaled "itrainedthismuch"
# needs to happen after desert removal for precip condition
# now deserts are also filled with -20 but doesn't matter since outside landmask and therfore ignored
logging.info('normalise precipitation ...')
log_fracmis(dataxr, 'before logscale precip')
dataxr = logscale_precip(dataxr)
log_fracmis(dataxr, 'after logscale precip')


### FORTH PART: ADD FEATURES ###
# add latitude and longitude as predictors
if ilatlondata:
    logging.info('add latitude and longitude as predictors ...')
    tmp = timeinvariant.to_dataset(dim='variable')
    londata, latdata = np.meshgrid(timeinvariant.longitude, timeinvariant.latitude)
    tmp['latdata'] = (('latitude', 'longitude'), latdata)
    tmp['londata'] = (('latitude', 'longitude'), londata)
    timeinvariant = tmp.to_array()
    log_fracmis(dataxr_filled, 'after adding latlon')


# add time as predictor
# see related article here:
# https://towardsdatascience.com/how-not-to-use-machine-learning-for-time-series-forecasting-avoiding-the-pitfalls-19f9d7adf424
# should it learn to predict the difference instead?
if itimedata: 
    logging.info('add time as predictor ...')
    _, ntimesteps, nlandpoints = dataxr_filled.shape
    tmp = dataxr_filled.to_dataset(dim='variable')
    timedat = np.arange(ntimesteps)
    timedat = np.tile(timedat, nlandpoints).reshape(nlandpoints,*timedat.shape).T
    tmp['timedat'] = (('time','landpoints'), timedat)
    dataxr_filled = tmp.to_array()
    log_fracmis(dataxr_filled, 'time adding')

if itimelag:
    logging.info('add timelag as predictor ...')

    tmp = dataxr_filled.sel(variable=varnames).rolling(time=7, center=False, min_periods=1).mean()
    tmp = tmp.assign_coords(time=[time + np.timedelta64(7,'D') for time in tmp.coords['time'].values])
    tmp = tmp.assign_coords(variable=[f'{var}lag_7ff' for var in varnames])
    dataxr_lost = xr.concat([dataxr_lost, tmp], dim='variable', join='left', fill_value=0)
    dataxr_filled = xr.concat([dataxr_filled, tmp], dim='variable', join='left', fill_value=0)

    tmp = dataxr_filled.sel(variable=varnames).rolling(time=7, center=False, min_periods=1).mean()
    tmp = tmp.assign_coords(variable=[f'{var}lag_7' for var in varnames])
    dataxr_lost = xr.concat([dataxr_lost, tmp], dim='variable', join='left')
    dataxr_filled = xr.concat([dataxr_filled, tmp], dim='variable', join='left')

    tmp = dataxr_filled.sel(variable=varnames).rolling(time=30-7, center=False, min_periods=1).mean()
    tmp = tmp.assign_coords(time=[time + np.timedelta64(7,'D') for time in tmp.coords['time'].values])
    tmp = tmp.assign_coords(variable=[f'{var}lag_30' for var in varnames])
    dataxr_lost = xr.concat([dataxr_lost, tmp], dim='variable', join='left')
    dataxr_filled = xr.concat([dataxr_filled, tmp], dim='variable', join='left')

    tmp = dataxr_filled.sel(variable=varnames).rolling(time=30*6-30, center=False, min_periods=1).mean()
    tmp = tmp.assign_coords(time=[time + np.timedelta64(30,'D') for time in tmp.coords['time'].values])
    tmp = tmp.assign_coords(variable=[f'{var}lag_180' for var in varnames])
    dataxr_lost = xr.concat([dataxr_lost, tmp], dim='variable', join='left') # or overwrite!
    dataxr_filled = xr.concat([dataxr_filled, tmp], dim='variable', join='left')

    varmeans = dataxr_filled.mean(dim=('time'))
    dataxr_filled = dataxr_filled.fillna(varmeans)
    log_fracmis(dataxr_filled, 'after timelags added')

# standardise / normalise data
if inormalise:
    dataxr_filled = normalise(dataxr_filled, isave=True)
    timeinvariant = normalise(timeinvariant)

#  stack var and invar
logging.info(f'before stack timeinvar and var...')
ntimesteps = dataxr_filled.coords['time'].size
timeinvariant = np.repeat(timeinvariant, ntimesteps, axis=1)
timeinvariant['time'] = dataxr_filled['time'] # timeinvariant needs timestep for concat to work
dataxr_filled = xr.concat([dataxr_filled, timeinvariant], dim='variable')

# stack time and space
log_fracmis(dataxr_filled, 'before stack time and space ...')
dataxr_filled = dataxr_filled.stack(datapoints=('time','landpoints')).reset_index('datapoints').T.to_dataset(name='data')
lostmask = lostmask.stack(datapoints=('time','landpoints')).reset_index('datapoints').T.to_dataset(name='data')

# save data
logging.info(f'save data ...')
dataxr_filled.to_netcdf(netcdfpath + f'features_init_{missingness}_{frac_missing}_idebug_{idebugspace}.nc')
lostmask.to_netcdf(netcdfpath + f'lostmask_init_{missingness}_{frac_missing}_idebug_{idebugspace}.nc')
logging.info('DONE')
