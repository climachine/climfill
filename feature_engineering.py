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

def logscale_precip(data, varname='tp'):
    data.loc[varname] = data.loc[varname].where(data.loc[varname] > 0.000001, -1) # define lower threshold for "no rain"
    data.loc[varname] = xu.log(data.loc[varname])
    data.loc[varname] = data.loc[varname].where(~np.isnan(data.loc[varname]), -20) # all zero precip got -1, all -1 got nan, all nan get -20
    return data

def create_precip_binary(data, varname='tp'):
    ip = (data.loc[varname] < 0.00001)*1
    ip = ip.rename('ip')
    return ip

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
    return data.stack(datapoints=('time','landpoints')).reset_index('datapoints').T.to_dataset(name='data')

def create_lat_lon_features(constant_maps):
    londata, latdata = np.meshgrid(constant_maps.longitude, constant_maps.latitude)
    latitude_arr = (('latitude', 'longitude'), latdata)
    longitude_arr = (('latitude', 'longitude'), londata)
    return latitude_arr, longitude_arr

def create_time_feature(data):
    _, ntimesteps, nlandpoints = data.shape
    timedat = np.arange(ntimesteps)
    timedat = np.tile(timedat, nlandpoints).reshape(nlandpoints,*timedat.shape).T
    time_arr = (('time','landpoints'), timedat)
    return time_arr

def create_embedded_features(data, s, l):
    """
    window size s
    time lag l
    """

    # rolling window average
    tmp = data.sel(variable=varnames).rolling(time=l-s, center=False, min_periods=1).mean()

    # overwrite time stamp to current day
    tmp = tmp.assign_coords(time=[time + np.timedelta64(s,'D') for time in tmp.coords['time'].values])

    # rename feature to not overwrite variable
    tmp = tmp.assign_coords(variable=[f'{var}lag_{s}ff' for var in varnames])

    # fill missing values in lagged features at beginning or end of time series
    varmeans = tmp.mean(dim=('time'))
    tmp = tmp.fillna(varmeans)

    return tmp

    
if __name__ == '__main__':
    data = xr.open_dataset('/path/to/gappy/dataset')
    mask = xr.open_dataset('/path/to/mask/dataset')

    ip = create_precip_binary(data, varname='tp')
    data = data.to_dataset(dim='variable')
    data['ip'] = ip
    data = data.to_array()
    data = logscale_precip(data, varname='tp')

data = xr.concat([data, tmp], dim='variable', join='left', fill_value=0)

# steps of the feature engineering

# log-scale precip DONE

# add latitude, longitude, time DONE

# add lags DONE

# normalise DONE

# stack constant maps and variables

# stack to datapoints

# divide precip into bool "itrained" and logscaled "itrainedthismuch"
# needs to happen after desert removal for precip condition
# now deserts are also filled with -20 but doesn't matter since outside landmask and therfore ignored
logging.info('normalise precipitation ...')
log_fracmis(dataxr, 'before logscale precip')
dataxr = logscale_precip(dataxr)
log_fracmis(dataxr, 'after logscale precip')


tmp = timeinvariant.to_dataset(dim='variable')
tmp['latdata'] = latitude_arr
tmp['londata'] = longitude_arr
tmp['timedat'] = time_arr
timeinvariant = tmp.to_array()


# standardise / normalise data
dataxr_filled = normalise(dataxr_filled, isave=True)
timeinvariant = normalise(timeinvariant)

#  stack var and invar
def stack_constant_maps(data, constant_maps):
    ntimesteps = data.coords['time'].size
    constant_maps = np.repeat(constant_maps, ntimesteps, axis=1)
    constant_maps['time'] = data['time'] # timeinvariant needs timestep for concat to work
    return constant_maps

data = xr.concat([data, constant_maps], dim='variable')

